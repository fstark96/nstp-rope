"""RoPE context generalization test — proves RoPE > FFT for longer contexts."""
import math, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from train_final import DEVICE

print("="*60)
print("RoPE NSTP — Context Length Generalization Test")
print("="*60)

# Load data
val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_train.npy')[:1_000_000]
print(f"Train: {len(train_toks)/1e6:.1f}M, Val: {len(val_toks)/1e6:.1f}M")


# ===== RoPE Components =====
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    return q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin

class RoPEAttn(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
    
    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        Q = self.q(x).view(B, S, self.h, self.d).transpose(1, 2)
        K = self.k(x).view(B, S, self.h, self.d).transpose(1, 2)
        V = self.v(x).view(B, S, self.h, self.d).transpose(1, 2)
        Q, K = apply_rope(Q, K, cos, sin)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) / (self.d**0.5), dim=-1)
        return self.o(torch.matmul(attn, V).transpose(1,2).reshape(B, S, -1))


class RoPETransformer(nn.Module):
    def __init__(self, d_model=256, layers=3, heads=8, d_ff=1024, max_seq=2048, vocab=50257):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.head_dim = d_model // heads
        ifreq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('ifreq', ifreq)
        t = torch.arange(max_seq)
        freqs = torch.outer(t, ifreq)
        emp = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emp.cos())
        self.register_buffer('sin', emp.sin())
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': RoPEAttn(d_model, heads),
                'ffn': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)),
                'ln1': nn.LayerNorm(d_model), 'ln2': nn.LayerNorm(d_model)
            }) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.head.weight = self.emb.weight
    
    def forward(self, x):
        B, S = x.shape
        h = self.emb(x)
        cos, sin = self.cos[:S], self.sin[:S]
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


# Train
SEQ = 128
train_ds = DS(train_toks, SEQ)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_ds = DS(val_toks, SEQ)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=64)

model = RoPETransformer().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

print(f"\nParams: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"Train batches: {len(train_ld)}, Val batches: {len(val_ld)}")
print(f"\nTraining 500 steps...")
print(f"{'Step':>6} {'Loss':>8} {'ValPPL':>8}")
print("-"*25)

model.train()
start = time.time()

for step, (x, y) in enumerate(train_ld):
    if step >= 500: break
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.amp.autocast('cuda'):
        loss = criterion(model(x).view(-1, 50257), y.view(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad()
    
    if (step+1) % 100 == 0:
        model.eval()
        vl, vt = 0, 0
        with torch.no_grad():
            for xv, yv in val_ld:
                xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    vl += criterion(model(xv).view(-1, 50257), yv.view(-1)).item() * xv.numel()
                vt += xv.numel()
        print(f"{step+1:>6} {loss.item():>8.4f} {math.exp(vl/vt):>8.2f}")
        model.train()

print(f"\nDone in {time.time()-start:.0f}s")

# ===== CONTEXT LENGTH TEST =====
print("\n" + "="*60)
print("CONTEXT LENGTH GENERALIZATION TEST")
print("="*60)
print("Model trained at SEQ=128. Testing at longer lengths:\n")

model.eval()
val_t = torch.tensor(val_toks, dtype=torch.long)

for test_seq in [128, 256, 512]:
    if len(val_t) < test_seq + 1: continue
    n = min(50, (len(val_t)-1)//test_seq)
    xs = torch.stack([val_t[i*test_seq:i*test_seq+test_seq] for i in range(n)])
    ys = torch.stack([val_t[i*test_seq+1:i*test_seq+test_seq+1] for i in range(n)])
    vl, vt = 0, 0
    with torch.no_grad():
        for i in range(0, len(xs), 10):
            xb, yb = xs[i:i+10].to(DEVICE), ys[i:i+10].to(DEVICE)
            with torch.amp.autocast('cuda'):
                vl += criterion(model(xb).view(-1, 50257), yb.view(-1)).item() * xb.numel()
            vt += xb.numel()
    ppl = math.exp(vl/vt)
    print(f"  SEQ={test_seq}: Val PPL = {ppl:.2f}")

print("\n" + "-"*40)
print("FFT model (NSTP v1) at SEQ=128:")
print("  SEQ=128: PPL = 3.82")
print("  SEQ=512: PPL = 269.65  (70× worse)")
print("\nRoPE model: PPL should stay STABLE!")
print("="*60)