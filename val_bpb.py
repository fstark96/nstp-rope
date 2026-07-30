"""
val_bpb.py — Vocab-independent bits-per-byte evaluation for NSTP-Ω V3.

Why val_bpb instead of perplexity?
- PPL is biased by vocab size (larger vocab → higher PPL even if model is "better")
- BPB normalizes by token byte-length: measures "how many bits does it take to
  predict the next byte?"
- Makes architectural changes fairly comparable across configs

Formula:
    val_bpb = total_nats / (ln(2) * total_bytes)
    where total_nats = sum of per-token CE losses (excluding special tokens)
    and total_bytes = sum of byte lengths of target tokens (excluding special tokens)
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional


# Approximate byte lengths for GPT-2 BPE tokens (cl1008k base).
# For FineWeb-Edu tokenized with GPT-2 tokenizer.
# This is approximate — exact mapping requires the actual tokenizer vocab.
# Special tokens (<|endoftext|>, etc.) have 0 bytes.
def estimate_token_bytes(vocab_size: int = 50257) -> torch.Tensor:
    """
    Return a (vocab_size,) tensor with estimated byte length per token.

    For GPT-2 tokenizer:
    - Token 0-256: typically 1 byte (single chars)
    - Most tokens: 1-4 bytes (BPE subwords)
    - Special tokens (50256): 0 bytes

    This is approximate but sufficient for relative BPB comparison.
    """
    # Default: most tokens ~2 bytes
    byte_lengths = torch.ones(vocab_size, dtype=torch.float32) * 2.0

    # First 256 tokens are typically single chars (1 byte)
    byte_lengths[:256] = 1.0

    # Larger BPE merges → more bytes
    # Tokens 256-1000: ~2-3 bytes
    byte_lengths[256:1000] = 2.5
    byte_lengths[1000:5000] = 3.0
    byte_lengths[5000:20000] = 3.5
    byte_lengths[20000:50000] = 4.0
    byte_lengths[50000:] = 4.5

    # Special tokens (GPT-2): <|endoftext|> = 50256, has 0 bytes
    if vocab_size > 50256:
        byte_lengths[50256] = 0.0

    return byte_lengths


def compute_val_bpb(
    model,
    val_loader,
    device: torch.device,
    vocab_size: int = 50257,
    token_bytes: Optional[torch.Tensor] = None,
    max_eval_batches: int = 50,
) -> float:
    """
    Compute validation bits-per-byte.

    Args:
        model: NSTP-Ω model
        val_loader: yields (x, y) batches
        device: torch device
        vocab_size: vocab size
        token_bytes: optional (vocab_size,) tensor with byte length per token
                    if None, uses estimate_token_bytes
        max_eval_batches: max batches to evaluate

    Returns:
        val_bpb: bits per byte (lower = better)
    """
    if token_bytes is None:
        token_bytes = estimate_token_bytes(vocab_size)
    token_bytes = token_bytes.to(device)

    model.eval()
    total_nats = 0.0
    total_bytes = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_eval_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x, return_drafts=False)
            logits = out['logits']

            # Per-token CE in nats
            loss_flat = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='none'
            )
            y_flat = y.view(-1)

            # Get byte lengths for each target token
            nbytes = token_bytes[y_flat]

            # Exclude special tokens (byte_length = 0)
            mask = nbytes > 0

            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()

    model.train()

    if total_bytes == 0:
        return float('inf')

    # Convert nats/byte to bits/byte (multiply by 1/ln(2))
    val_bpb = total_nats / (math.log(2) * total_bytes)
    return val_bpb


def compute_val_ppl(
    model,
    val_loader,
    device: torch.device,
    vocab_size: int = 50257,
    max_eval_batches: int = 50,
) -> float:
    """
    Compute validation perplexity (existing V2 metric, kept for comparison).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_eval_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x, return_drafts=False)
            logits = out['logits']

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    model.train()

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    return ppl


if __name__ == "__main__":
    print("Testing val_bpb evaluation...")

    from nstp_omega import NSTPOmega, NSTPOmegaConfig

    torch.manual_seed(0)
    config = NSTPOmegaConfig(vocab_size=1000, d_model=128, num_layers=2, num_heads=2)
    model = NSTPOmega(config).cuda()

    # Fake data
    val_data = torch.randint(0, 1000, (4, 128))
    val_target = torch.randint(0, 1000, (4, 128))

    # Convert to loader-like iterator
    class FakeLoader:
        def __init__(self, x, y, n_batches=5):
            self.x, self.y = x, y
            self.n = n_batches
        def __iter__(self):
            for i in range(self.n):
                yield self.x, self.y

    loader = FakeLoader(val_data.cuda(), val_target.cuda(), n_batches=5)

    # Test BPB
    bpb = compute_val_bpb(model, loader, torch.device('cuda'), vocab_size=1000, max_eval_batches=5)
    print(f"  Val BPB: {bpb:.4f}")

    # Test PPL
    ppl = compute_val_ppl(model, loader, torch.device('cuda'), vocab_size=1000, max_eval_batches=5)
    print(f"  Val PPL: {ppl:.2f}")

    print("\n✅ val_bpb evaluation working!")
