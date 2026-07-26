"""Quick diagnostic: load checkpoint and measure PPL with exact same code as train_final.py"""
import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, math, numpy as np, torch.nn as nn, torch.nn.functional as F, sys
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE

DEVICE = torch.device('cuda')
SEQ, BATCH, VS = 128, 8, 50257

val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')

class LMDataset:
    def __init__(self, toks, seq):
        t = torch.tensor(toks, dtype=torch.long)
        n = max(0, (len(t) - 1) // seq)
        self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
        self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

val_ds = LMDataset(val_toks, SEQ)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, num_workers=0)

# Same model def as train_final.py
class VH:
    @staticmethod
    def bind(h, pos, hsa_dim):
        freq = torch.fft.rfft(h, dim=-1)
        n = freq.shape[-1]; f = torch.arange(n, device=h.device, dtype=h.dtype)
        angle = 2 * math.pi * f * pos.float().unsqueeze(-1) / hsa_dim
        rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * rot, n=hsa_dim, dim=-1)
    @staticmethod
    def unbind(M, pos, hsa_dim):
        B, S = pos.shape
        M_exp = M.unsqueeze(1).expand(-1, S, -1)
        freq = torch.fft.rfft(M_exp, dim=-1)
        n = freq.shape[-1]; f = torch.arange(n, device=M.device, dtype=M.dtype)
        angle = 2 * math.pi * f * (-pos.float()).unsqueeze(-1) / hsa_dim
        rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * rot, n=hsa_dim, dim=-1)

class Enc(nn.Module):
    def __init__(self, d_model, hsa_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x): return F.normalize(self.proj(x), p=2, dim=-1)

class HDAAttn(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1, denoise_iter=3):
        super().__init__()
        self.dm = d_model; self.hd = hsa_dim; self.nh = num_heads
        self.head_dim = hsa_dim // num_heads
        self.encoders = nn.ModuleList([Enc(d_model, self.head_dim) for _ in range(num_heads)])
        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=denoise_iter, binary=False)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hsa_dim, d_model); self.drop = nn.Dropout(dropout)

    def forward(self, x, positions):
        heads = []
        for h in range(self.nh):
            h_enc = self.encoders[h](x)
            h_bound = VH.bind(h_enc, positions, self.head_dim)
            M = h_bound.mean(dim=1)  # [B, head_dim]
            h_ret = VH.unbind(M, positions, self.head_dim)
            h_ret = self.denoisers[h](h_ret)
            heads.append(h_ret)
        combined = torch.cat(heads, dim=-1)
        out = self.out_proj(combined)
        return self.drop(out), None

class NSTPBlock(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                 router_tt_ranks, expert_tt_ranks, dropout=0.1):
        super().__init__()
        self.attn = HDAAttn(d_model, hsa_dim, num_heads, dropout)
        self.moe = TTCERMoE(
            d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
            router_tt_ranks=router_tt_ranks, expert_tt_ranks=expert_tt_ranks,
            activation='gelu', dropout=dropout, router_aux_loss_coef=0.01,
        )
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, positions):
        r = x; x = self.norm1(x); a, _ = self.attn(x, positions); x = r + self.drop(a)
        r = x; x = self.norm2(x); m, _ = self.moe(x); return r + m

class NSTPModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hsa_dim,
                 num_experts, top_k, d_ff, router_tt_ranks, expert_tt_ranks, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            NSTPBlock(d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                      router_tt_ranks, expert_tt_ranks, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, ids, positions=None):
        B, S = ids.shape; dev = ids.device
        if positions is None:
            positions = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.drop(self.embed(ids))
        for block in self.blocks: x = block(x, positions)
        return self.head(self.norm(x))

model = NSTPModel(50257, 320, 3, 4, 2048, 4, 2, 768, [1,4,4,1], [1,4,4,4,1], 0.1).to(DEVICE)
sd = torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models/final_best.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(sd)
model.eval()
print(f'Loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

# Quick test: forward pass
x0 = val_ds.xs[:2].to(DEVICE)
pos = torch.arange(SEQ, device=DEVICE).unsqueeze(0).expand(2, -1)
with torch.no_grad():
    out = model(x0, pos)
    print(f'Output: {out.shape}, mean={out.mean().item():.4f}, max={out.max().item():.2f}')

# Compute ppl the exact way train_final.py does
crit = nn.CrossEntropyLoss(reduction='mean')
total_loss, total_tokens = 0.0, 0
model.eval()
with torch.no_grad():
    for x, y in val_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)  # positions=None → auto-generated
        loss = crit(out.view(-1, 50257), y.view(-1))
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
ppl = math.exp(total_loss / total_tokens)
print(f'\nVal PPL: {ppl:.2f}')
print(f'CE: {total_loss/total_tokens:.4f}')
print(f'Random baseline: {math.exp(math.log(50257)):.0f}')