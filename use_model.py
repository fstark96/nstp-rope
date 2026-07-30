"""
use_model.py — Load NSTP-Ω V2 checkpoint and use it.

After V2 training completes, run:
    python use_model.py --checkpoint models_v2/checkpoints/nstp_omega_step050000.pt \
                        --prompt "The meaning of life is" \
                        --max-tokens 100 \
                        --temperature 0.8 \
                        --top-k 50

Modes:
    generate  - Sample text from a prompt (default)
    perplexity - Compute PPL on a text file
    embed     - Extract embeddings for downstream tasks
    info      - Print model info, no generation
"""
import os
import sys
import math
import argparse
from pathlib import Path

REPO_DIR = Path('C:/Users/user/AppData/Local/Temp/nstp-v2')
PYTHON_EXE = r'C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'

# Fix profile module conflict
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
import sys as _sys
_sys.modules['profile'] = FakeProfile()


def get_default_checkpoint():
    """Find the highest-step V2 checkpoint."""
    ckpt_dir = REPO_DIR / 'models_v2' / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob('nstp_omega_step*.pt'),
                   key=lambda p: int(p.stem.split('step')[-1] or 0),
                   reverse=True)
    return str(ckpts[0]) if ckpts else None


def load_model_and_tokenizer(checkpoint_path: str):
    """Load NSTPOmega from checkpoint + GPT-2 tokenizer."""
    sys.path.insert(0, str(REPO_DIR))
    from nstp_omega import NSTPOmega, NSTPOmegaConfig

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Detect config from state dict
    state = ckpt.get('model', ckpt)
    config_keys = set()
    for key in state.keys():
        parts = key.split('.')
        if 'embed' in parts[0]:
            config_keys.add('embed')
            break

    # Use default V2 config (matches our training)
    config = NSTPOmegaConfig(
        vocab_size=50257,
        d_model=512,
        num_layers=6,
        num_heads=8,
        num_experts=4,
        head_dim=64,
        hhm_l2_dim=2048,
        hhm_l3_dim=8192,
        hhm_num_prototypes=512,
        tt_ranks=[2, 4, 8, 2],
        dropout=0.0,  # No dropout for inference
        layer_drop=0.0,
        min_layers=3,
        max_layers=6,
        halt_threshold=1.0,
        target_sparsity=1.0,
    )

    model = NSTPOmega(config)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {total/1e6:.1f}M params")

    # Tokenizer (GPT-2 BPE via tiktoken)
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')

    return model, enc, config


def generate(model, enc, prompt: str, max_tokens: int = 100,
             temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9):
    """Generate text from prompt."""
    import torch
    import torch.nn.functional as F

    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = [50256] + tokens  # BOS
    ids = torch.tensor([tokens], dtype=torch.long,
                       device=next(model.parameters()).device)

    print(f"\nPrompt: {prompt!r}")
    print(f"  ({len(tokens)} tokens)")
    print(f"\n--- Generating ({max_tokens} tokens) ---\n")

    generated_ids = []
    with torch.no_grad():
        for step in range(max_tokens):
            # Forward (limit context to 2048 to avoid OOM)
            logits = model(ids[:, -2048:])['logits']
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                mask = cumprobs > top_p
                mask[:, 1:] = mask[:, :-1].clone()  # Keep first above threshold
                mask[:, 0] = False
                sorted_logits[mask] = -float('inf')
                next_logits = torch.zeros_like(next_logits).scatter(
                    -1, sorted_idx, sorted_logits
                )

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

            # Stop on EOS
            if next_id.item() == 50256:
                break

            ids = torch.cat([ids, next_id], dim=1)
            generated_ids.append(next_id.item())

            # Print incrementally (for long generations)
            if step % 20 == 0:
                chunk = enc.decode(generated_ids[-20:])
                print(chunk, end='', flush=True)

        print()  # Newline at end

    # Decode full generation
    output_text = enc.decode(generated_ids)
    print(f"\n\n--- Full Output ---\n")
    print(prompt + output_text)
    print()

    return output_text


def compute_perplexity(model, enc, text_path: str):
    """Compute PPL on a text file."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    print(f"Computing PPL on: {text_path}")
    with open(text_path) as f:
        text = f.read()

    tokens = enc.encode(text)
    print(f"  {len(tokens):,} tokens")

    # Convert to tensor
    device = next(model.parameters()).device
    tokens_t = torch.tensor([tokens], dtype=torch.long, device=device)

    # Compute in chunks
    chunk_size = 512
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens) - chunk_size, chunk_size):
            end = start + chunk_size + 1
            chunk = tokens_t[:, start:end]

            x = chunk[:, :-1]
            y = chunk[:, 1:]
            logits = model(x)['logits']

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity:  {ppl:.2f}")
    return ppl


def main():
    parser = argparse.ArgumentParser(description='Use NSTP-Ω V2 trained model')
    parser.add_argument('mode', choices=['generate', 'perplexity', 'info'],
                        default='generate', nargs='?')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: latest V2 checkpoint)')
    parser.add_argument('--prompt', type=str, default='The meaning of life is',
                        help='Generation prompt')
    parser.add_argument('--text', type=str, default=None,
                        help='Text file for perplexity mode')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    args = parser.parse_args()

    # Find checkpoint
    checkpoint = args.checkpoint or get_default_checkpoint()
    if not checkpoint:
        print("ERROR: No checkpoint found in models_v2/checkpoints/")
        print("       Wait for V2 training to complete step 50K.")
        sys.exit(1)

    print("=" * 70)
    print(f"NSTP-Ω V2 — MODE: {args.mode.upper()}")
    print("=" * 70)

    if args.mode == 'info':
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        print(f"\nCheckpoint: {checkpoint}")
        if 'step' in ckpt:
            print(f"  Step: {ckpt['step']}")
        if 'best_val' in ckpt:
            print(f"  Best Val PPL: {ckpt['best_val']:.2f}")
        state = ckpt.get('model', ckpt)
        total = sum(v.numel() for v in state.values() if torch.is_tensor(v))
        print(f"  Total params: {total/1e6:.1f}M")
        return

    # Load model + tokenizer
    import torch
    model, enc, config = load_model_and_tokenizer(checkpoint)

    if args.mode == 'generate':
        generate(model, enc, args.prompt,
                 max_tokens=args.max_tokens,
                 temperature=args.temperature,
                 top_k=args.top_k,
                 top_p=args.top_p)
    elif args.mode == 'perplexity':
        if not args.text:
            print("ERROR: --text required for perplexity mode")
            sys.exit(1)
        compute_perplexity(model, enc, args.text)


if __name__ == '__main__':
    import torch
    main()
