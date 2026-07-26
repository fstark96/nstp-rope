import sys
sys.modules['profile'] = type(sys)('fp')
import torch, time, os
VOCAB=50257; D_MODEL=512; NUM_HEADS=8; HEAD_DIM=64; NUM_LAYERS=6; D_FF=2048; MAX_SEQ=2048

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)
class RoPE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ifreq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer('ifreq', ifreq)
        t = torch.arange(MAX_SEQ); freqs = torch.outer(t, ifreq)
        emp = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emp.cos()); self.register_buffer('sin', emp.sin())
    def forward(self, Q, K):
        S = Q.size(2); cos = self.cos[:S].unsqueeze(0).unsqueeze(0); sin = self.sin[:S].unsqueeze(0).unsqueeze(0)
        return Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, D_MODEL); self.drop = torch.nn.Dropout(0.1)
        self.rope = RoPE()
        self.layers = torch.nn.ModuleList([self._make_layer() for _ in range(NUM_LAYERS)])
        self.ln_f = torch.nn.LayerNorm(D_MODEL); self.head = torch.nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight
    def _make_layer(self):
        return torch.nn.ModuleDict({
            'q_proj': torch.nn.Linear(D_MODEL, D_MODEL), 'k_proj': torch.nn.Linear(D_MODEL, D_MODEL),
            'v_proj': torch.nn.Linear(D_MODEL, D_MODEL), 'o_proj': torch.nn.Linear(D_MODEL, D_MODEL),
            'ffn': torch.nn.Sequential(torch.nn.Linear(D_MODEL, D_FF), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(D_FF, D_MODEL)),
            'ln1': torch.nn.LayerNorm(D_MODEL), 'ln2': torch.nn.LayerNorm(D_MODEL),
        })
    def forward(self, x):
        B, S = x.shape; h = self.drop(self.emb(x))
        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        for L in self.layers:
            q = L['q_proj'](L['ln1'](h)).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = L['k_proj'](L['ln1'](h)).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = L['v_proj'](L['ln1'](h)).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            q, k = self.rope(q, k)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False, dropout_p=0.0)
            h = h + L['o_proj'](attn.transpose(1,2).reshape(B, S, -1)); h = h + L['ffn'](L['ln2'](h))
        return self.head(self.ln_f(h))

model = Model().cuda().half()
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {params:.1f}M params")

configs = [(8,128), (4,256), (4,512), (2,1024)]
for bs, seq in configs:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    xb = torch.randint(0, VOCAB, (bs, seq)).cuda()
    t0 = time.time()
    try:
        with torch.amp.autocast('cuda'):
            out = model(xb); loss = out.sum()
        loss.backward(); torch.cuda.synchronize()
        ms = (time.time()-t0)*1000
        peak = torch.cuda.max_memory_allocated()/1e9
        print(f"  bs={bs}, seq={seq}: {ms:.0f}ms, peak={peak:.2f}GB")
    except Exception as e:
        print(f"  bs={bs}, seq={seq}: OOM")
    del xb, out, loss; model.zero_grad()