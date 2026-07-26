"""Progressive RoPE on WikiText-2: 3 stages (SEQ=128→256→512).
Tests whether RoPE actually generalizes to unseen context lengths.

Stages:
  S0: 10K steps @ SEQ=128
  S1:  5K steps @ SEQ=256 (fine-tune from S0)
  S2:  5K steps @ SEQ=512 (fine-tune from S1)

After each stage we eval PPL at 128/256/512 to measure generalization.
"""
import os
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_USE_NEPER'] = '0'

import sys
class FakeProfile:
    def run(self, *a, **k): pass
    def runctx(self, *a, **k): pass
sys.modules['profile'] = FakeProfile()

import math, time, torch, torch.nn as nn, numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
D_MODEL  = 256
NUM_HEADS = 8
HEAD_DIM  = D_MODEL // NUM_HEADS   # 32
NUM_LAYERS = 2
D_FF      = 768
VOCAB     = 50257
MAX_SEQ   = 2048
DEVICE    = torch.device('cuda')

# ── data ──────────────────────────────────────────────────────────────────────
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')
print(f"WikiText-2: Train={len(train_toks)/1e6:.2f}M  Val={len(val_toks)/1e6:.2f}M  Test={len(test_toks)/1e6:.2f}M")

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
        """Q,K: (B, h, S, hd) → rotated Q,K"""
        S = Q.size(2)
        cos = self.cos[:S].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(0)
        return Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin

# ── model ─────────────────────────────────────────────────────────────────────
class RoPEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.drop = nn.Dropout(0.1)
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
                nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(D_FF, D_MODEL)),
            'ln1': nn.LayerNorm(D_MODEL),
            'ln2': nn.LayerNorm(D_MODEL),
        })

    def forward(self, x):
        B, S = x.shape
        h = self.drop(self.emb(x))
        for L in self.layers:
            # Project to Q,K,V
            q = L['q_proj'](L['ln1'](h))
            k = L['k_proj'](L['ln1'](h))
            v = L['v_proj'](L['ln1'](h))
            # Reshape: (B, S, h, hd) → (B, h, S, hd)
            q = q.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            # Apply RoPE
            q, k = self.rope(q, k)
            # Causal attention
            scale = HEAD_DIM ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            h = h + L['o_proj'](torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1))
            # FFN
            h = h + L['ffn'](L['ln2'](h))
        return self.head(self.ln_f(h))

# ── dataset ───────────────────────────────────────────────────────────────────
class SimpleDS:
    def __init__(self, tokens, seq_len, max_tokens=None):
        if max_tokens:
            tokens = tokens[:max_tokens]
        t = torch.tensor(tokens, dtype=torch.long)
        n = max(0, (len(t) - 1) // seq_len)
        self.xs = torch.stack([t[i*seq_len:i*seq_len+seq_len] for i in range(n)])
        self.ys = torch.stack([t[i*seq_len+1:i*seq_len+seq_len+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

# ── eval ──────────────────────────────────────────────────────────────────────
def eval_ppl(model, tokens, seq, bs=16):
    if len(tokens) < seq + 1:
        return None
    n = min(200, (len(tokens)-1)//seq)
    xs = torch.stack([torch.tensor(tokens[i*seq:i*seq+seq]) for i in range(n)]).long()
    ys = torch.stack([torch.tensor(tokens[i*seq+1:i*seq+seq+1]) for i in range(n)]).long()
    total_loss, total_toks = 0.0, 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            xb = xs[i:i+bs].cuda()
            yb = ys[i:i+bs].cuda()
            with torch.amp.autocast('cuda'):
                out = model(xb)
                total_loss += crit(out.view(-1, VOCAB), yb.view(-1)).item() * xb.numel()
            total_toks += xb.numel()
    return math.exp(total_loss / total_toks)

# ── train one stage ───────────────────────────────────────────────────────────
def train_stage(model, opt, tokens, seq_len, steps, lr, eval_every=2000, stage=""):
    model.train()
    ds = SimpleDS(tokens, seq_len)
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
                                     drop_last=True, num_workers=0)
    opt.param_groups[0]['lr'] = lr
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    step = 0
    t0 = time.time()
    iter_ld = iter(ld)

    while step < steps:
        try:
            x, y = next(iter_ld)
        except StopIteration:
            iter_ld = iter(ld)
            x, y = next(iter_ld)
        x, y = x.cuda(), y.cuda()
        with torch.amp.autocast('cuda'):
            loss = crit(model(x).view(-1, VOCAB), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()
        step += 1
        if step % eval_every == 0:
            elapsed = time.time() - t0
            model.eval()
            r128 = eval_ppl(model, val_toks, 128)
            r256 = eval_ppl(model, val_toks, 256)
            r512 = eval_ppl(model, val_toks, 512)
            print(f"  {stage} {step}: S128={r128:.2f}  S256={r256:.2f}  S512={r512:.2f}  ({elapsed:.0f}s)")
            model.train()
    return step

# ── main ──────────────────────────────────────────────────────────────────────
print("="*70)
print("Progressive RoPE — WikiText-2  (128 → 256 → 512)")
print("="*70)

model = RoPEModel().cuda()
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {params:.1f}M params  (d={D_MODEL}, L={NUM_LAYERS}, h={NUM_HEADS}, d_ff={D_FF})")

crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

# Speed check
test_ds = SimpleDS(train_toks, 128, max_tokens=200_000)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=32, num_workers=0)
xb, yb = next(iter(test_ld)); xb, yb = xb.cuda(), yb.cuda()
with torch.amp.autocast('cuda'):
    loss = crit(model(xb).view(-1, VOCAB), yb.view(-1))
loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_test = torch.optim.AdamW(model.parameters(), lr=1e-3); opt_test.step(); opt_test.zero_grad()
t0 = time.time()
for _ in range(30):
    with torch.amp.autocast('cuda'):
        loss = crit(model(xb).view(-1, VOCAB), yb.view(-1))
    loss.backward(); opt_test.step(); opt_test.zero_grad()
ms = (time.time()-t0)/30*1000
print(f"Speed: {ms:.1f}ms/step")
print(f"  S0 (10K): {ms*10000/60000:.0f}min  |  S1 (5K): {ms*5000/60000:.0f}min  |  S2 (5K): {ms*5000/60000:.0f}min")
print(f"  TOTAL est: {(ms*(10000+5000+5000))/60000:.0f}min")
del opt_test

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

# STAGE 0: SEQ=128
print("\n" + "="*70)
print("STAGE 0 — SEQ=128 (10K steps)")
print("="*70)
train_stage(model, opt, train_toks, 128, 10_000, 1e-3, eval_every=2000, stage="S0")
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_wt2_s0.pt')
model.eval()
print("  After S0 (trained at 128 only):")
for s in [128, 256, 512]:
    p = eval_ppl(model, val_toks, s)
    print(f"    S{s}={p:.2f}")

# STAGE 1: SEQ=256
print("\n" + "="*70)
print("STAGE 1 — Fine-tune at SEQ=256 (5K steps)")
print("="*70)
train_stage(model, opt, train_toks, 256, 5_000, 5e-4, eval_every=2000, stage="S1")
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_wt2_s1.pt')
model.eval()
print("  After S1 (seen 256):")
for s in [128, 256, 512]:
    p = eval_ppl(model, val_toks, s)
    print(f"    S{s}={p:.2f}")

# STAGE 2: SEQ=512
print("\n" + "="*70)
print("STAGE 2 — Fine-tune at SEQ=512 (5K steps)")
print("="*70)
train_stage(model, opt, train_toks, 512, 5_000, 2e-4, eval_every=2000, stage="S2")
torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_wt2_s2.pt')
model.eval()
print("  After S2 (seen 512):")
for s in [128, 256, 512]:
    p = eval_ppl(model, val_toks, s)
    print(f"    S{s}={p:.2f}")

# Test eval
print("\n" + "="*70)
print("TEST SET")
print("="*70)
for name, path, seq in [("S0", "rope_wt2_s0.pt", 128), ("S1", "rope_wt2_s1.pt", 256), ("S2", "rope_wt2_s2.pt", 512)]:
    model.load_state_dict(torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models/' + path, weights_only=True))
    model.eval()
    p = eval_ppl(model, test_toks, seq)
    print(f"  {name} @ SEQ={seq}: Test PPL = {p:.2f}")

print("\n✓ Complete")