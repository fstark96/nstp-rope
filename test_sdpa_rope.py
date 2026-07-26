import sys
sys.modules['profile'] = type(sys)('fp')
import torch, torch.nn as nn, math, time

VOCAB=50257; D_MODEL=384; NUM_HEADS=6; HEAD_DIM=64; NUM_LAYERS=6; D_FF=1536; MAX_SEQ=2048

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
        S = Q.size(2)
        cos = self.cos[:S].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(0)
        return Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin

class Model(nn.Module):
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
            'q_proj': nn.Linear(D_MODEL, D_MODEL), 'k_proj': nn.Linear(D_MODEL, D_MODEL),
            'v_proj': nn.Linear(D_MODEL, D_MODEL), 'o_proj': nn.Linear(D_MODEL, D_MODEL),
            'ffn': nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(0.1), nn.Linear(D_FF, D_MODEL)),
            'ln1': nn.LayerNorm(D_MODEL), 'ln2': nn.LayerNorm(D_MODEL),
        })
    def forward(self, x):
        B, S = x.shape
        h = self.drop(self.emb(x))
        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        for L in self.layers:
            q = L['q_proj'](L['ln1'](h)).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = L['k_proj'](L['ln1'](h)).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = L['v_proj'](L['ln1'](h)).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            q, k = self.rope(q, k)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, is_causal=False, dropout_p=0.0)
            h = h + L['o_proj'](attn.transpose(1,2).reshape(B, S, -1))
            h = h + L['ffn'](L['ln2'](h))
        return self.head(self.ln_f(h))

model = Model().cuda()
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {params:.1f}M params")

crit = nn.CrossEntropyLoss()
for seq in [128, 256, 512, 1024]:
    xb = torch.randint(0, VOCAB, (16, seq)).cuda()
    yb = torch.randint(0, VOCAB, (16, seq)).cuda()
    with torch.amp.autocast('cuda'):
        out = model(xb)
        loss = crit(out.view(-1, VOCAB), yb.view(-1))
    loss.backward(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.amp.autocast('cuda'):
            out = model(xb)
            loss = crit(out.view(-1, VOCAB), yb.view(-1))
        loss.backward()
    torch.cuda.synchronize()
    ms = (time.time()-t0)/10*1000
    ok = "OK" if (not torch.isnan(loss) and not torch.isinf(loss)) else "FAIL"
    print(f"  SEQ={seq}: {ms:.1f}ms/step  loss={loss.item():.4f}  [{ok}]")
    model.zero_grad()

print()
print("Learning test (100 steps):")
x = torch.randint(0, VOCAB, (32, 128)).cuda()
y = torch.randint(0, VOCAB, (32, 128)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(100):
    with torch.amp.autocast('cuda'):
        out = model(x)
        loss = crit(out.view(-1, VOCAB), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
status = "learning" if loss.item() < 4.0 else "check"
print(f"  After 100 steps: loss={loss.item():.4f}  [{status}]")
print()
print("SDPA + RoPE: WORKING" if loss.item() < 4.0 else "WARNING: loss still high")