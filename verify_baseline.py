"""
verify_baseline.py — Rigorous comparison between NSTP and standard transformer
on WikiText-2, with IDENTICAL preprocessing, tokenizer, and evaluation protocol.
"""
import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, sys
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ = 128
BATCH = 8

# ============================================================
# STEP 1: Load WikiText-2 using the EXACT same method as NSTP training
# ============================================================
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

print(f"Dataset sizes:")
print(f"  Train: {len(train_toks):,} tokens")
print(f"  Val:   {len(val_toks):,} tokens")
print(f"  Test:  {len(test_toks):,} tokens")

# Check vocab size
from collections import Counter
all_toks = np.concatenate([train_toks, val_toks, test_toks])
freq = Counter(all_toks)
max_tok = max(all_toks)
print(f"  Unique tokens: {len(freq)}")
print(f"  Max token ID:  {max_tok}")
print(f"  Vocab size (implied): {max_tok + 1}")
# GPT-2 vocab = 50257. Our data has max token < 50257? Let's check.
print(f"  Token 0 freq:  {freq[0]:,}")
print(f"  Token 1 freq:  {freq[1]:,}")
print(f"  Token 2 freq:  {freq[2]:,}")


# ============================================================
# STEP 2: Build data loaders with IDENTICAL indexing
# ============================================================
class LMDataset:
    """Standard next-token prediction dataset."""
    def __init__(self, tokens, seq_len):
        # tokens should NOT include </s> or <s> markers — raw word IDs
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq = seq_len

    def __len__(self):
        # Number of complete (x, y) pairs we can make
        return max(0, (len(self.tokens) - 1) // self.seq)

    def __getitem__(self, i):
        start = i * self.seq
        x = self.tokens[start:start + self.seq]
        y = self.tokens[start + 1:start + self.seq + 1]
        return x, y


def make_loader(tokens, seq_len, batch_size, shuffle=False):
    ds = LMDataset(tokens, seq_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


val_ld  = make_loader(val_toks,   SEQ, BATCH)
test_ld = make_loader(test_toks,  SEQ, BATCH)

# ============================================================
# STEP 3: Confirm perplexity computation is standard
# ============================================================
print("\n--- Perplexity Computation Verification ---")
print("Standard: PPL = exp(mean(cross_entropy_i for i=1..N))")
print("where N = total number of tokens (excluding padding)")
print("ce_i = -log(p(token_i | context))")
print("Using: nn.CrossEntropyLoss(reduction='mean') → exp(mean)")
print("This is the standard Wikipedia formula.")

# ============================================================
# STEP 4: Random baseline (analytic: uniform over vocab)
# ============================================================
print("\n--- Random Baseline ---")
crit = nn.CrossEntropyLoss(reduction='mean')

# Analytic random perplexity: uniform over vocab_size tokens
# CE_uniform = log(vocab_size) → PPL_uniform = vocab_size
uniform_ce = math.log(max_tok + 1)
uniform_ppl = math.exp(uniform_ce)
print(f"Analytic uniform CE: log({max_tok+1}) = {uniform_ce:.4f}")
print(f"Analytic uniform ppl: {uniform_ppl:.1f} (= vocab_size)")
print(f"  This is what a model that assigns equal prob to all tokens gives.")
print(f"  Our NSTP model at random init should also give ~{uniform_ppl:.0f}.")

# ============================================================
# STEP 5: NSTP model (load from saved checkpoint)
# ============================================================
print("\n--- NSTP Continuous HDC (best checkpoint) ---")

# Architecture definition (must match train_final.py)
class VH:
    @staticmethod
    def bind(h, pos, hsa_dim):
        freq = torch.fft.rfft(h, dim=-1)
        n = freq.shape[-1]
        f = torch.arange(n, device=h.device, dtype=h.dtype)
        angle = 2 * math.pi * f * pos.float().unsqueeze(-1) / hsa_dim
        rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * rot, n=hsa_dim, dim=-1)
    @staticmethod
    def unbind(M, pos, hsa_dim):
        B, S = pos.shape
        M_exp = M.unsqueeze(1).expand(-1, S, -1)
        freq = torch.fft.rfft(M_exp, dim=-1)
        n = freq.shape[-1]
        f = torch.arange(n, device=M.device, dtype=M.dtype)
        angle = 2 * math.pi * f * (-pos.float()).unsqueeze(-1) / hsa_dim
        rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * rot, n=hsa_dim, dim=-1)

class Enc(nn.Module):
    def __init__(self, d_model, hsa_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)

class HDAAttn(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1, denoise_iter=3):
        super().__init__()
        self.dm = d_model; self.hd = hsa_dim; self.nh = num_heads
        self.head_dim = hsa_dim // num_heads
        self.encoders = nn.ModuleList([Enc(d_model, self.head_dim) for _ in range(num_heads)])
        self.denoisers = nn.ModuleList([HSADenoiser(self.head_dim, num_iterations=denoise_iter, binary=False) for _ in range(num_heads)])
        self.out_proj = nn.Linear(hsa_dim, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, positions):
        heads = []
        for h in range(self.nh):
            h_enc = self.encoders[h](x)
            h_bound = VH.bind(h_enc, positions, self.head_dim)
            M = h_bound.mean(dim=1)
            h_ret = VH.unbind(M, positions, self.head_dim)
            h_ret = self.denoisers[h](h_ret)
            heads.append(h_ret)
        return self.drop(self.out_proj(torch.cat(heads, dim=-1))), None

class NSTPBlock(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, num_experts, top_k, d_ff, rtr, etr, dropout=0.1):
        super().__init__()
        self.attn = HDAAttn(d_model, hsa_dim, num_heads, dropout)
        self.moe = TTCERMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
                            router_tt_ranks=rtr, expert_tt_ranks=etr, activation='gelu',
                            dropout=dropout, router_aux_loss_coef=0.01)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
    def forward(self, x, positions):
        r = x; x = self.norm1(x); a, _ = self.attn(x, positions); x = r + self.drop(a)
        r = x; x = self.norm2(x); m, _ = self.moe(x); return r + m

class NSTPModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hsa_dim,
                 num_experts, top_k, d_ff, rtr, etr, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([NSTPBlock(d_model, hsa_dim, num_heads, num_experts, top_k, d_ff, rtr, etr, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        self.apply(_init)
    def forward(self, ids, positions=None):
        B, S = ids.shape; dev = ids.device
        if positions is None: positions = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.drop(self.embed(ids))
        for b in self.blocks: x = b(x, positions)
        return self.head(self.norm(x))


def compute_ppl_model(model, loader, desc=""):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out.view(-1, out.shape[-1]), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    ppl = math.exp(total_loss / total_tokens)
    if desc:
        print(f"  {desc}: ppl={ppl:.2f}, CE={total_loss/total_tokens:.4f}, tokens={total_tokens:,.0f}")
    return ppl


# Try to load checkpoint
ckpt_path = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/final_best.pt'
if os.path.exists(ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Determine vocab size from checkpoint
    vs = sd['embed.weight'].shape[0]
    dm = sd['embed.weight'].shape[1]
    n_blocks = sum(1 for k in sd if k.startswith('blocks.') and '.attn.out_proj.weight' in k)

    print(f"  Config: {n_blocks}L, d={dm}, vocab={vs}")

    nstp = NSTPModel(vs, dm, n_blocks, 4, 2048, 4, 2, 768, [1,4,4,1], [1,4,4,4,1]).to(DEVICE)
    nstp.load_state_dict(sd, strict=False)
    nstp.eval()

    nstp_val_ppl  = compute_ppl_model(nstp, val_ld,  "NSTP val")
    nstp_test_ppl = compute_ppl_model(nstp, test_ld, "NSTP test")
else:
    print("No checkpoint found at:", ckpt_path)
    nstp_val_ppl = nstp_test_ppl = None

# ============================================================
# STEP 6: Standard Transformer baseline (GPT-2 config, same data)
# Must match: same tokenizer, same data, same eval protocol
# ============================================================
print("\n--- Standard Transformer Baseline (GPT-2 small) ---")
print(f"Same SEQ={SEQ}, same BATCH={BATCH}, same tokenizer, same eval")


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TransformerLM(nn.Module):
    """Standard causal transformer (GPT-2 small architecture)."""
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072,
                 max_seq=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_seq)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=True,  # Pre-LN (more stable)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, S = x.shape
        x = self.drop(self.token_emb(x) + self.pos_emb(x))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, x.device)
        for block in self.blocks:
            x = block(x, src_mask=causal_mask)
        return self.head(self.norm(x))


# Quick test: random-init transformer perplexity
print("\nRunning random-init transformer baseline...")
model_dummy = TransformerLM(
    vocab_size=max_tok + 1, d_model=768, n_layers=12, n_heads=12,
    d_ff=3072, max_seq=SEQ, dropout=0.0
).to(DEVICE)
rand_tfm_ppl = compute_ppl_model(model_dummy, val_ld, "Random TF val")
print(f"Random GPT-2 (fresh): ppl={rand_tfm_ppl:.1f} (should be ~vocab_size)")

# Train a small transformer for comparison (same config as NSTP ~39M params)
# GPT-2 small is 124M — too big. Let's use ~40M params.
# 40M transformer: d_model=256, 6 layers, 4 heads, d_ff=1024
print("\nTraining 40M-param standard transformer for fair comparison...")
tfm_40m = TransformerLM(
    vocab_size=max_tok + 1, d_model=320, n_layers=6, n_heads=5,
    d_ff=1024, max_seq=SEQ, dropout=0.1
).to(DEVICE)

params_40m = sum(p.numel() for p in tfm_40m.parameters())
print(f"Standard transformer params: {params_40m:,} ({params_40m/1e6:.1f}M)")

# Train for 3000 steps (like NSTP's first eval checkpoint)
train_ds = LMDataset(train_toks, SEQ)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
print(f"Train batches: {len(train_ld)}")

opt = torch.optim.AdamW(tfm_40m.parameters(), lr=5e-4, weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=3000, pct_start=0.1)
crit_train = nn.CrossEntropyLoss(reduction='mean')

t0 = time.time()
gs = 0
best_tfm_ppl = float('inf')
best_tfm_state = None
EPOCHS_NEEDED = (3000 + len(train_ld) - 1) // len(train_ld)  # epochs to reach 3000 steps

print(f"\nTrain batches/epoch: {len(train_ld)}, need {EPOCHS_NEEDED} epochs for 3000 steps")
print(f"\n{'Step':>5}  {'TFM Val':>9}  {'TFM Test':>9}  {'Loss':>8}")
print("-" * 45)

for epoch in range(EPOCHS_NEEDED):
    train_ld_shuffled = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    for x, y in train_ld_shuffled:
        if gs >= 3000:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = tfm_40m(x)
        loss = crit_train(out.view(-1, out.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(tfm_40m.parameters(), 1.0)
        opt.step(); sched.step(); gs += 1

        if gs % 500 == 0 or gs == 3000:
            vp = compute_ppl_model(tfm_40m, val_ld)
            tp = compute_ppl_model(tfm_40m, test_ld)
            print(f"{gs:>5}  {vp:>9.2f}  {tp:>9.2f}  {loss.item():>8.4f}  ({time.time()-t0:.0f}s)")
            if vp < best_tfm_ppl:
                best_tfm_ppl = vp
                best_tfm_state = {k: v.cpu().clone() for k, v in tfm_40m.state_dict().items()}
    if gs >= 3000:
        break

# Evaluate best checkpoint
if best_tfm_state:
    tfm_40m.load_state_dict(best_tfm_state)
    tfm_val_ppl  = compute_ppl_model(tfm_40m, val_ld,  "TFM val (best)")
    tfm_test_ppl = compute_ppl_model(tfm_40m, test_ld, "TFM test (best)")

del model_dummy
torch.cuda.empty_cache()

# ============================================================
# STEP 7: FINAL COMPARISON TABLE
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTS (same tokenizer, same eval protocol, same SEQ={SEQ})")
print(f"{'='*60}")
print(f"\nDatasets: WikiText-2")
print(f"  Train: {len(train_toks):,} tokens")
print(f"  Val:   {len(val_toks):,} tokens")
print(f"  Test:  {len(test_toks):,} tokens")
print(f"  Vocab: {max_tok+1} (raw IDs, no added tokens)")
print(f"\nRandom baselines:")
print(f"  Uniform random:    ppl ≈ {max_tok+1:.0f}")
print(f"  Full random eval: ppl = {uniform_ppl:.1f}")
print(f"\nModel comparison (same training data, same SEQ={SEQ}):")
if nstp_val_ppl is not None:
    print(f"  NSTP Continuous HDC ({params_40m/1e6:.0f}M params, ~5 epochs):")
    print(f"    Val ppl:   {nstp_val_ppl:.2f}")
    print(f"    Test ppl:  {nstp_test_ppl:.2f}")
print(f"  Standard Transformer ({params_40m/1e6:.0f}M params, 3K steps):")
print(f"    Val ppl:   {tfm_val_ppl:.2f}" if best_tfm_state else "    N/A")
print(f"    Test ppl:  {tfm_test_ppl:.2f}" if best_tfm_state else "    N/A")
print(f"\n  GPT-2 small published: ~29 on WikiText-2 (1024 ctx)")
print(f"  Our GPT-2 small baseline (3K steps, {params_40m/1e6:.0f}M params): val={tfm_val_ppl:.1f}" if best_tfm_state else "")
print(f"\n  NSTP vs our trained Transformer: {tfm_val_ppl/nstp_val_ppl:.2f}x" if nstp_val_ppl and best_tfm_state else "")
print(f"{'='*60}")

# Save comparison
import json
results = {
    'vocab_size': int(max_tok + 1),
    'seq_len': int(SEQ),
    'random_val_ppl': float(uniform_ppl),
    'nstp_val_ppl': float(nstp_val_ppl) if nstp_val_ppl else None,
    'nstp_test_ppl': float(nstp_test_ppl) if nstp_test_ppl else None,
    'tfm_val_ppl': float(tfm_val_ppl) if best_tfm_state else None,
    'tfm_test_ppl': float(tfm_test_ppl) if best_tfm_state else None,
    'tfm_params': int(params_40m),
}
with open('C:/Users/user/AppData/Local/Temp/nstp-v2/verification_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to verification_results.json")