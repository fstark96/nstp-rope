import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, math, sys
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE


def vectorized_hadamard_bind(h: torch.Tensor, pos: torch.Tensor, hsa_dim: int) -> torch.Tensor:
    """FFT-based circular convolution binding for continuous hypervectors."""
    freq = torch.fft.rfft(h, dim=-1)
    freqs = torch.arange(freq.shape[-1], device=h.device, dtype=h.dtype)
    pos_float = pos.float()
    angle = 2 * np.pi * freqs * pos_float.unsqueeze(-1) / hsa_dim
    pos_freq = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    pos_freq = torch.complex(pos_freq[..., 0], pos_freq[..., 1])
    bound_freq = freq * pos_freq
    return torch.fft.irfft(bound_freq, n=hsa_dim, dim=-1)


def vectorized_hadamard_unbind(M: torch.Tensor, pos: torch.Tensor, hsa_dim: int) -> torch.Tensor:
    """Unbind: retrieve context vector at positions. M: [batch, hsa_dim], pos: [batch, seq]"""
    batch, seq = pos.shape
    # M: [batch, hsa_dim] -> [batch, 1, hsa_dim] for irfft
    M_exp = M.unsqueeze(1).expand(-1, seq, -1)  # [batch, seq, hsa_dim]
    freq = torch.fft.rfft(M_exp, dim=-1)  # [batch, seq, freq_dim]
    freq_dim = freq.shape[-1]
    freqs = torch.arange(freq_dim, device=M.device, dtype=M.dtype)
    pos_float = pos.float()  # [batch, seq]
    angle = 2 * np.pi * freqs * (-pos_float).unsqueeze(-1) / hsa_dim
    pos_freq = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    pos_freq = torch.complex(pos_freq[..., 0], pos_freq[..., 1])
    unbind_freq = freq * pos_freq
    return torch.fft.irfft(unbind_freq, n=hsa_dim, dim=-1)


class ContinuousHDCEncoder(nn.Module):
    """Learned projection to continuous hypervectors (L2 normalized)."""

    def __init__(self, d_model: int, hsa_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return F.normalize(h, p=2, dim=-1)


class ContinuousHDAAttention(nn.Module):
    """Continuous HDC Attention — fully vectorized."""

    def __init__(self, d_model: int, hsa_dim: int = 4096, num_heads: int = 4,
                 denoise_iterations: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hsa_dim = hsa_dim
        self.num_heads = num_heads
        self.head_dim = hsa_dim // num_heads

        self.encoders = nn.ModuleList([
            ContinuousHDCEncoder(d_model, self.head_dim) for _ in range(num_heads)
        ])

        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=denoise_iterations, binary=False)
            for _ in range(num_heads)
        ])

        self.output_proj = nn.Linear(hsa_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, return_context: bool = False):
        batch, seq, _ = x.shape

        head_outputs = []
        for h_idx in range(self.num_heads):
            h_enc = self.encoders[h_idx](x)
            h_bound = vectorized_hadamard_bind(h_enc, positions, self.head_dim)
            M = h_bound.mean(dim=1)
            retrieved = vectorized_hadamard_unbind(M, positions, self.head_dim)
            retrieved = self.denoisers[h_idx](retrieved)
            head_outputs.append(retrieved)

        combined = torch.cat(head_outputs, dim=-1)
        out = self.output_proj(combined)
        out = self.dropout(out)

        return out, None


class ContinuousNSTPBlock(nn.Module):

    def __init__(self, d_model: int, hsa_dim: int, num_heads: int,
                 num_experts: int, top_k: int, d_ff: int,
                 router_tt_ranks, expert_tt_ranks, dropout: float = 0.1):
        super().__init__()

        self.attention = ContinuousHDAAttention(
            d_model=d_model, hsa_dim=hsa_dim, num_heads=num_heads, dropout=dropout
        )

        self.moe = TTCERMoE(
            d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
            router_tt_ranks=router_tt_ranks, expert_tt_ranks=expert_tt_ranks,
            activation="gelu", dropout=dropout, router_aux_loss_coef=0.01,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, positions)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        moe_out, _ = self.moe(x)
        x = residual + moe_out

        return x


class ContinuousNSTPModel(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_layers: int,
                 num_heads: int, hsa_dim: int, num_experts: int, top_k: int,
                 d_ff: int, router_tt_ranks, expert_tt_ranks, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ContinuousNSTPBlock(
                d_model=d_model, hsa_dim=hsa_dim, num_heads=num_heads,
                num_experts=num_experts, top_k=top_k, d_ff=d_ff,
                router_tt_ranks=router_tt_ranks, expert_tt_ranks=expert_tt_ranks,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Use untied weights — embedding and lm_head are separate
        # Tying causes shared-parameter bug that destroys learning
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)
        print(f"ContinuousNSTP initialized:")
        print(f"  d_model={d_model}, hsa_dim={hsa_dim}")
        print(f"  num_layers={num_layers}, num_experts={num_experts}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
        batch, seq = input_ids.shape
        device = input_ids.device

        if positions is None:
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)

        x = self.embedding(input_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, positions)

        x = self.norm(x)
        return self.lm_head(x)


# ---- Benchmark ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, SEQ = 4, 128
LR = 5e-4
STEPS = 10000
EVAL_EVERY = 1000

train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')

class DS:
    def __init__(self, toks, seq):
        self.toks = torch.tensor(toks, dtype=torch.long)
        self.seq = seq
    def __len__(self): return max(0, len(self.toks) // self.seq)
    def __getitem__(self, i):
        s = self.toks[i*self.seq:(i+1)*self.seq+1]
        return s[:-1], s[1:]

train_ds = DS(train_toks, SEQ)
val_ds   = DS(val_toks,   SEQ)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH, num_workers=0)

def train_model(name, model, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    crit = nn.CrossEntropyLoss()

    def ppl(loader):
        model.eval()
        loss, tok = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss += crit(out.view(-1, 50257), y.view(-1)).item() * x.numel()
                tok += x.numel()
        return math.exp(loss / tok)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"{name}: {params:,} params ({params/1e6:.1f}M)")
    print(f"{'='*60}")

    t0 = time.time()
    gs = 0
    best_val = float('inf')

    for x, y in train_ld:
        if gs >= steps: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = crit(out.view(-1, 50257), y.view(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        gs += 1
        if gs % EVAL_EVERY == 0:
            vp = ppl(val_ld)
            elapsed = time.time() - t0
            print(f"  Step {gs:5d}: val_ppl={vp:.1f}  ({elapsed:.0f}s)")
            if vp < best_val: best_val = vp

    print(f"  Final (step {gs}): best_val_ppl={best_val:.1f}")
    return best_val

print(f"Device: {DEVICE}")
print(f"Train batches: {len(train_ld)}, Val batches: {len(val_ld)}")

# Continuous HDC
model_cont = ContinuousNSTPModel(
    vocab_size=50257, d_model=320, num_layers=3, num_heads=4,
    hsa_dim=2048, num_experts=4, top_k=2, d_ff=768,
    router_tt_ranks=[1, 4, 4, 1], expert_tt_ranks=[1, 4, 4, 4, 1],
    dropout=0.1
).to(DEVICE)

val_cont = train_model("CONTINUOUS HDC", model_cont)

print(f"\n{'='*60}")
print(f"RESULTS (step 2000):")
print(f"  FIXED binary encoder:     ~1767 (7.5M params)")
print(f"  TRAINABLE binary:         ~1773 (7.5M params)")
print(f"  CONTINUOUS HDC:            {val_cont:.1f}")
print(f"  GPT-2 small baseline:      ~29")
if val_cont < 1767:
    print(f"  → CONTINUOUS is {1767/val_cont:.2f}x better than binary")
else:
    print(f"  → CONTINUOUS is {val_cont/1767:.2f}x WORSE than binary")