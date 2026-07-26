"""Progressive RoPE training — optimized for speed.
Stage 1: SEQ=128, 5M tokens
Stage 2: SEQ=512, 10M tokens (jump test — does RoPE generalize from 128 to 512?)"""
import os
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_USE_NEPER'] = '0'

import math, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from train_final import DEVICE

print("="*60)
print("Progressive RoPE — 2 Stage (128 → 512)")
print("="*60)

val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_train.npy')
print(f"Train: {len(train_toks)/1e6:.1f}M, Val: {len(val_toks)/1e6:.1f}M")

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

class RoPEAttn(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads; self.d = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        Q = self.q(x).view(B, S, self.h, self.d).transpose(1,2)
        K = self.k(x).view(B, S, self.h, self.d).transpose(1,2)
        V = self.v(x).view(B, S, self.h, self.d).transpose(1,2)
        cos = cos.unsqueeze(0).unsqueeze(0); sin = sin.unsqueeze(0).unsqueeze(0)
        Q = Q * cos + rotate_half(Q) * sin
        K = K * cos + rotate_half(K) * sin
        scale = self.d ** -0.5
        attn = torch.matmul(Q, K.transpose(-2,-1)) * scale
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        return self.o(torch.matmul(attn, V).transpose(1,2).reshape(B, S, -1))

class RoPETransformer(nn.Module):
    def __init__(self, d_model=192, layers=2, heads=8, d_ff=768, max_seq=2048, vocab=50257):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.hd = d_model // heads
        ifreq = 1.0 / (10000 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer('ifreq', ifreq)
        t = torch.arange(max_seq); freqs = torch.outer(t, ifreq)
        emp = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emp.cos()); self.register_buffer('sin', emp.sin())
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': RoPEAttn(d_model, heads),
                'ffn': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)),
                'ln1': nn.LayerNorm(d_model), 'ln2': nn.LayerNorm(d_model)
            }) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab); self.head.weight = self.emb.weight
    def forward(self, x):
        B, S = x.shape
        h = self.emb(x); cos, sin = self.cos[:S], self.sin[:S]
        for l in self.layers:
            h = h + l['attn'](l['ln1'](h), cos, sin)
            h = h + l['ffn'](l['ln2'](h))
        return self.head(self.ln_f(h))

class DS:
    def __init__(self, toks, seq):
        t = torch.tensor(toks, dtype=torch.long)
        n = max(0, (len(t)-1)//seq)
        self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
        self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

def eval_at_lengths(model, val_toks, lengths, batch_size=16):
    crit = nn.CrossEntropyLoss()
    val_t = torch.tensor(val_toks, dtype=torch.long)
    results = {}
    for seq in lengths:
        if len(val_t) < seq + 1: continue
        n = min(50, (len(val_t)-1)//seq)
        xs = torch.stack([val_t[i*seq:i*seq+seq] for i in range(n)])
        ys = torch.stack([val_t[i*seq+1:i*seq+seq+1] for i in range(n)])
        vl, vt = 0, 0
        with torch.no_grad():
            for i in range(0, len(xs), batch_size):
                xb = xs[i:i+batch_size].to(DEVICE)
                yb = ys[i:i+batch_size].to(DEVICE)
                with torch.amp.autocast('cuda'):
                    vl += crit(model(xb).view(-1, 50257), yb.view(-1)).item() * xb.numel()
                vt += xb.numel()
        results[seq] = math.exp(vl/vt)
    return results

model = RoPETransformer().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

total_params = sum(p.numel() for p in model.parameters())/1e6
print(f"Model: {total_params:.1f}M params")

# Quick timing test
print("\nTiming test...")
train_ds_128 = DS(train_toks[:5_000_000], 128)
train_ld_128 = torch.utils.data.DataLoader(train_ds_128, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
xb, yb = next(iter(train_ld_128))
xb, yb = xb.to(DEVICE), yb.to(DEVICE)
t0 = time.time()
for _ in range(50):
    with torch.amp.autocast('cuda'):
        loss = crit(model(xb).view(-1, 50257), yb.view(-1))
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad()
ms_per_step = (time.time()-t0)/50*1000
print(f"{ms_per_step:.1f}ms/step, {ms_per_step*3000/60:.0f}min for 3000 steps")
batches_s1 = len(train_ld_128)
print(f"Stage 1: {batches_s1} batches ({batches_s1/1000:.1f}K), {ms_per_step*batches_s1/60:.0f}min")

# Stage 2 timing
train_ds_512 = DS(train_toks[:10_000_000], 512)
train_ld_512 = torch.utils.data.DataLoader(train_ds_512, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
xb2, yb2 = next(iter(train_ld_512))
xb2, yb2 = xb2.to(DEVICE), yb2.to(DEVICE)
t0 = time.time()
for _ in range(10):
    with torch.amp.autocast('cuda'):
        loss = crit(model(xb2).view(-1, 50257), yb2.view(-1))
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad()
ms_per_step_512 = (time.time()-t0)/10*1000
batches_s2 = len(train_ld_512)
print(f"Stage 2: {batches_s2} batches ({batches_s2/1000:.1f}K), {ms_per_step_512*batches_s2/60:.0f}min")

print(f"\n{'='*60}")
print(f"ESTIMATED TOTAL: {(ms_per_step*batches_s1 + ms_per_step_512*batches_s2)/60:.0f}min")
print(f"{'='*60}")

# ===== STAGE 1: SEQ=128 =====
print(f"\n{'='*60}")
print("STAGE 1: SEQ=128")
print(f"{'='*60}")
train_ds = DS(train_toks[:5_000_000], 128)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
print(f"Batches: {len(train_ld)}, LR: 1e-3")

model.train()
step = 0; t0 = time.time()
for x, y in train_ld:
    if step >= 3000: break
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.amp.autocast('cuda'):
        loss = crit(model(x).view(-1, 50257), y.view(-1))
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
    if step % 500 == 0:
        elapsed = time.time()-t0
        model.eval()
        r = eval_at_lengths(model, val_toks, [128, 512])
        row = "  ".join(f"S{s}:{v:.2f}" for s,v in r.items())
        print(f"  Step {step}: {row} ({elapsed:.0f}s)")
        model.train()

# ===== STAGE 2: SEQ=512 =====
print(f"\n{'='*60}")
print("STAGE 2: SEQ=512")
print(f"{'='*60}")
for g in opt.param_groups: g['lr'] = 3e-4

train_ds2 = DS(train_toks[:10_000_000], 512)
train_ld2 = torch.utils.data.DataLoader(train_ds2, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
print(f"Batches: {len(train_ld2)}, LR: 3e-4")

model.train()
step = 0; t0 = time.time()
for x, y in train_ld2:
    if step >= 3000: break
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.amp.autocast('cuda'):
        loss = crit(model(x).view(-1, 50257), y.view(-1))
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
    if step % 500 == 0:
        elapsed = time.time()-t0
        model.eval()
        r = eval_at_lengths(model, val_toks, [128, 512])
        row = "  ".join(f"S{s}:{v:.2f}" for s,v in r.items())
        print(f"  Step {step}: {row} ({elapsed:.0f}s)")
        model.train()

# ===== FINAL RESULTS =====
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
model.eval()
results = eval_at_lengths(model, val_toks, [128, 256, 512], batch_size=16)
for seq, ppl in sorted(results.items()):
    print(f"  SEQ={seq}: PPL = {ppl:.2f}")

torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
           'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_progressive.pt')
print(f"\nSaved to models/rope_progressive.pt")