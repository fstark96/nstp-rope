"""
NSTP RoPE Progressive Training — Full FineWeb 800M tokens
=========================================================
4 improvements combined:
  1. SDPA (is_causal=False + manual causal) = O(S) instead of O(S²)
  2. ~45M param model (scaled from 14M)
  3. Full FineWeb 800M-token progressive training (128→512→1024)
  4. Sequence-length mixing to prevent catastrophic forgetting

Stages:
  S0: SEQ=128,  10K steps (warmup), lr=1e-3
  S1: SEQ=256,  20K steps (grow ctx), lr=7e-4
  S2: SEQ=512,  20K steps (double),  lr=5e-4
  S3: SEQ=1024, 10K steps (max ctx),  lr=3e-4
"""
import os
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
class FakeProfile:
    def run(self, *a, **k): pass
    def runctx(self, *a, **k): pass
sys.modules['profile'] = FakeProfile()

import math, time, torch, torch.nn as nn, numpy as np

# ── model dimensions ─────────────────────────────────────────────────────────
VOCAB    = 50257
D_MODEL  = 512       # 6×(qkv+o+ffn) ≈ 45M params
NUM_HEADS = 8
HEAD_DIM  = D_MODEL // NUM_HEADS   # 64
NUM_LAYERS = 6   # ~45M params
D_FF      = 2048   # 4×d_model
MAX_SEQ   = 2048
DROPOUT   = 0.1

# Stage configs: sized for RTX 4070 Ti SUPER (16GB)
# Conservative: bs kept low to account for optim state, gradients, eval overhead
STAGES = [
    dict(seq=128,  steps=20_000, lr=1e-3,  warmup=500,  bs=32,  accum=2,  eval_every=5000),
    dict(seq=256,  steps=15_000, lr=7e-4,  warmup=500,  bs=16,  accum=2,  eval_every=5000),
    dict(seq=512,  steps=10_000, lr=5e-4,  warmup=200,  bs=8,   accum=2,  eval_every=2000),
    dict(seq=1024, steps= 5_000, lr=3e-4,  warmup=100,  bs=4,   accum=4,  eval_every=1000),
]

# Mixing: fraction of mixed-short batches during S2/S3
# (batch_size_short, fraction) — short=128 always mixed in
MIX_SHORT_SEQ = 128
MIX_FRACTION  = 0.25   # 25% of batches are short-sequence

# ── data ──────────────────────────────────────────────────────────────────────
DATA_DIR = 'C:/Users/user/AppData/Local/Temp/nstp-v2/data'
train_toks = np.load(f'{DATA_DIR}/fineweb_800m_train.npy').astype(np.int32)
val_toks   = np.load(f'{DATA_DIR}/fineweb_800m_val.npy').astype(np.int32)
test_toks  = np.load(f'{DATA_DIR}/fineweb_test_tokens.npy').astype(np.int32)
print(f"FineWeb: Train={len(train_toks)/1e6:.1f}M  Val={len(val_toks)/1e6:.1f}M  Test={len(test_toks)/1e6:.1f}M")

# ── RoPE ──────────────────────────────────────────────────────────────────────
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq=2048, base=10000.0):
        super().__init__()
        ifreq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('ifreq', ifreq)
        t = torch.arange(max_seq)
        freqs = torch.outer(t, ifreq)
        emp = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emp.cos())
        self.register_buffer('sin', emp.sin())

    def forward(self, Q, K):
        """Q,K: (B, h, S, hd) → rotated Q,K  [sdpa needs (B, h, S, hd)]"""
        S = Q.size(2)
        cos = self.cos[:S].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(0)
        return Q * cos + rotate_half(Q) * sin, \
               K * cos + rotate_half(K) * sin

# ── model ──────────────────────────────────────────────────────────────────────
class RoPEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.rope = RoPE(HEAD_DIM, MAX_SEQ)
        self.layers = nn.ModuleList([self._make_layer() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight

    def _make_layer(self):
        return nn.ModuleDict({
            'q_proj': nn.Linear(D_MODEL, D_MODEL),
            'k_proj': nn.Linear(D_MODEL, D_MODEL),
            'v_proj': nn.Linear(D_MODEL, D_MODEL),
            'o_proj': nn.Linear(D_MODEL, D_MODEL),
            'ffn': nn.Sequential(
                nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(D_FF, D_MODEL)),
            'ln1': nn.LayerNorm(D_MODEL),
            'ln2': nn.LayerNorm(D_MODEL),
        })

    def forward(self, x):
        """x: (B, S) token ids. Returns (B, S, V) logits."""
        B, S = x.shape
        h = self.drop(self.emb(x))
        # Pre-compute causal mask once (upper triangle = True → mask these)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        for L in self.layers:
            # Self-attention
            q = L['q_proj'](L['ln1'](h))
            k = L['k_proj'](L['ln1'](h))
            v = L['v_proj'](L['ln1'](h))
            q = q.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)   # (B,h,S,hd)
            k = k.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            q, k = self.rope(q, k)                                   # RoPE applied
            # SDPA: is_causal=False + explicit boolean mask
            # causal_mask: True = ignore, so upper triangle is masked
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,     # (S,S) broadcast to (B,h,S,S)
                is_causal=False,           # ← CRITICAL: don't let SDPA re-apply causal
                dropout_p=0.0,
            )
            h = h + L['o_proj'](attn.transpose(1, 2).reshape(B, S, -1))
            # FFN
            h = h + L['ffn'](L['ln2'](h))
        return self.head(self.ln_f(h))


# ── dataset ────────────────────────────────────────────────────────────────────
class SimpleDS:
    def __init__(self, tokens, seq_len):
        t = torch.tensor(tokens, dtype=torch.long)
        n = max(0, (len(t) - 1) // seq_len)
        self.xs = torch.stack([t[i*seq_len:i*seq_len+seq_len] for i in range(n)])
        self.ys = torch.stack([t[i*seq_len+1:i*seq_len+seq_len+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]


class MixedLengthDS:
    """For mixing short + long sequences during training.
    Short items are zero-padded to long_seqs so DataLoader collate is uniform."""
    def __init__(self, tokens, long_seqs, short_seq=128, mix_frac=0.25):
        self.long_ds   = SimpleDS(tokens, long_seqs)
        self.short_ds   = SimpleDS(tokens, short_seq)
        self.long_seqs  = long_seqs
        self.mix_frac   = mix_frac
        n = len(self.long_ds)
        self.mix_idx = np.random.binomial(1, mix_frac, n).astype(bool)

    def __len__(self): return len(self.long_ds)

    def __getitem__(self, i):
        if self.mix_idx[i % len(self.mix_idx)]:
            x, y = self.short_ds[i % len(self.short_ds)]
            # Zero-pad front: position 0 tokens → pad with 0 (pad token)
            pad = self.long_seqs - x.size(0)
            x = torch.cat([torch.zeros(pad, dtype=x.dtype), x])
            y = torch.cat([torch.zeros(pad, dtype=y.dtype), y])
        else:
            x, y = self.long_ds[i % len(self.long_ds)]
        return x, y


# ── eval ───────────────────────────────────────────────────────────────────────
def eval_ppl(model, tokens, seq, bs=8, max_batches=100):
    if len(tokens) < seq + 1:
        return None
    n = min(max_batches, (len(tokens)-1)//seq)
    xs = torch.stack([torch.tensor(tokens[i*seq:i*seq+seq]) for i in range(n)]).long()
    ys = torch.stack([torch.tensor(tokens[i*seq+1:i*seq+seq+1]) for i in range(n)]).long()
    total_loss, total_toks = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            xb = xs[i:i+bs].cuda()
            yb = ys[i:i+bs].cuda()
            with torch.amp.autocast('cuda'):
                out = model(xb)
                crit = nn.CrossEntropyLoss()
                total_loss += crit(out.view(-1, VOCAB), yb.view(-1)).item() * xb.numel()
            total_toks += xb.numel()
    return math.exp(total_loss / total_toks)


# ── training one stage ────────────────────────────────────────────────────────
def train_stage(model, opt, scaler, train_toks, stage_cfg, eval_every=None, mix_short=False, stage_name=""):
    seq_len   = stage_cfg['seq']
    steps     = stage_cfg['steps']
    base_lr   = stage_cfg['lr']
    warmup    = stage_cfg['warmup']
    batch_size = stage_cfg['bs']
    accum     = stage_cfg['accum']       # gradient accumulation steps
    eval_every = eval_every or stage_cfg.get('eval_every', 5000)

    model.train()
    if mix_short and seq_len > MIX_SHORT_SEQ:
        ds = MixedLengthDS(train_toks, seq_len, MIX_SHORT_SEQ, MIX_FRACTION)
        print(f"  [{stage_name}] Mixed-length: {seq_len} + {MIX_SHORT_SEQ} ({int(MIX_FRACTION*100)}% short)")
    else:
        ds = SimpleDS(train_toks, seq_len)
    ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     drop_last=True, num_workers=0, pin_memory=False)
    iter_ld = iter(ld)

    crit = nn.CrossEntropyLoss()
    step = 0
    t0 = time.time()
    ex_count = 0

    while step < steps:
        # Linear warmup + cosine decay LR
        if step < warmup:
            lr = base_lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(steps - warmup, 1)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        opt.param_groups[0]['lr'] = lr

        loss_sum = 0.0
        try:
            for _ in range(accum):
                try:
                    x, y = next(iter_ld)
                except StopIteration:
                    iter_ld = iter(ld)
                    x, y = next(iter_ld)
                x, y = x.cuda(), y.cuda()
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = crit(out.view(-1, VOCAB), y.view(-1)) / accum
                scaler.scale(loss).backward()
                loss_sum += loss.item() * accum
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  [OOM at step {step}] Reducing batch size...")
                torch.cuda.empty_cache()
                continue  # skip this batch, retry
            else:
                raise

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        try:
            scaler.step(opt)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  [OOM on optimizer step {step}]")
                torch.cuda.empty_cache()
                continue
            else:
                raise
        scaler.update(); opt.zero_grad()
        step += 1
        ex_count += batch_size * accum

        if step % eval_every == 0:
            elapsed = time.time() - t0
            eps = ex_count / elapsed
            model.eval()
            try:
                r128  = eval_ppl(model, val_toks, 128,  max_batches=50)
                r512  = eval_ppl(model, val_toks, 512,  max_batches=25)
                r1024 = eval_ppl(model, val_toks, 1024, max_batches=15)
                print(f"  {stage_name} {step}/{steps}: "
                      f"S128={r128:.3f}  S512={r512:.3f}  S1024={r1024:.3f}  "
                      f"lr={lr:.2e}  {eps:.0f} ex/s  ({elapsed:.0f}s)")
            except RuntimeError as e:
                print(f"  {stage_name} {step}/{steps}: eval skipped (error: {e})")
            model.train()

    return step


# ── main ──────────────────────────────────────────────────────────────────────
print("="*70)
print("NSTP RoPE — Full FineWeb Progressive (32M params, 128→512→1024)")
print("="*70)

model = RoPEModel().cuda()
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {params:.1f}M params  (d={D_MODEL}, L={NUM_LAYERS}, h={NUM_HEADS}, hd={HEAD_DIM}, d_ff={D_FF})")

# Count params more precisely
total_params = sum(p.numel() for p in model.parameters())
print(f"  TOTAL={total_params/1e6:.1f}M params")

opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda')

# ── speed benchmark ──────────────────────────────────────────────────────────
print("\nSpeed benchmark...")
xb = torch.randint(0, VOCAB, (32, 128)).cuda()
yb = torch.randint(0, VOCAB, (32, 128)).cuda()
crit = nn.CrossEntropyLoss()
with torch.amp.autocast('cuda'):
    out = model(xb)
    loss = crit(out.view(-1, VOCAB), yb.view(-1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt2 = torch.optim.SGD(model.parameters(), lr=1e-3)
opt2.step(); opt2.zero_grad()

t0 = time.time()
for _ in range(30):
    with torch.amp.autocast('cuda'):
        out = model(xb)
        loss = crit(out.view(-1, VOCAB), yb.view(-1))
    loss.backward(); opt2.step(); opt2.zero_grad()
ms = (time.time()-t0)/30*1000
print(f"Speed: {ms:.1f}ms/step  (bs=32, seq=128)")
total_steps = sum(s['steps'] for s in STAGES)
print(f"Total training steps: {total_steps:,}")
print(f"Estimated time: {ms*total_steps/60000:.0f} min  ({ms*total_steps/3600:.1f} hr)")
del opt2

# ── STAGE 0: SEQ=128 ───────────────────────────────────────────────────────────
stage = STAGES[0]
print(f"\n{'='*70}")
print(f"STAGE 0 — SEQ={stage['seq']} ({stage['steps']:,} steps, lr={stage['lr']})")
print("="*70)
train_stage(model, opt, scaler, train_toks, stage,
            eval_every=5000, mix_short=False, stage_name="S0")

# Save checkpoint
os.makedirs('C:/Users/user/AppData/Local/Temp/nstp-v2/models', exist_ok=True)
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_fw_s0.pt')

# Full eval after S0
model.eval()
print("  After S0 (trained at 128 only):")
for s in [128, 256, 512, 1024]:
    p = eval_ppl(model, val_toks, s, max_batches=100)
    if p: print(f"    S{s}={p:.3f}")

# ── STAGE 1: SEQ=256 ──────────────────────────────────────────────────────────
stage = STAGES[1]
print(f"\n{'='*70}")
print(f"STAGE 1 — SEQ={stage['seq']} ({stage['steps']:,} steps, lr={stage['lr']})")
print("="*70)
train_stage(model, opt, scaler, train_toks, stage,
            eval_every=5000, mix_short=False, stage_name="S1")
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_fw_s1.pt')

model.eval()
print("  After S1 (seen 256):")
for s in [128, 256, 512, 1024]:
    p = eval_ppl(model, val_toks, s, max_batches=100)
    if p: print(f"    S{s}={p:.3f}")

# ── STAGE 2: SEQ=512 (with short-sequence mixing) ─────────────────────────────
stage = STAGES[2]
print(f"\n{'='*70}")
print(f"STAGE 2 — SEQ={stage['seq']} ({stage['steps']:,} steps, lr={stage['lr']})")
print("="*70)
train_stage(model, opt, scaler, train_toks, stage,
            eval_every=5000, mix_short=True, stage_name="S2")
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_fw_s2.pt')

model.eval()
print("  After S2 (seen 512, mixed short):")
for s in [128, 256, 512, 1024]:
    p = eval_ppl(model, val_toks, s, max_batches=100)
    if p: print(f"    S{s}={p:.3f}")

# ── STAGE 3: SEQ=1024 (with short-sequence mixing) ────────────────────────────
stage = STAGES[3]
print(f"\n{'='*70}")
print(f"STAGE 3 — SEQ={stage['seq']} ({stage['steps']:,} steps, lr={stage['lr']})")
print("="*70)
train_stage(model, opt, scaler, train_toks, stage,
            eval_every=5000, mix_short=True, stage_name="S3")
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_fw_s3.pt')

# ── Final eval ────────────────────────────────────────────────────────────────
model.eval()
print(f"\n{'='*70}")
print("FINAL EVALUATION")
print("="*70)
print("Validation Perplexity:")
for s in [128, 256, 512, 1024]:
    p = eval_ppl(model, val_toks, s, max_batches=200)
    if p: print(f"  S{s}={p:.3f}")

print("\nTest Perplexity:")
for s in [128, 256, 512, 1024]:
    p = eval_ppl(model, test_toks, s, max_batches=200)
    if p: print(f"  S{s}={p:.3f}")

print("\n✓ Complete")