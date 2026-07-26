"""
train_final.py — Best architecture from self-evolution:
- Simple mean-based context (NO cosine sim retrieval — proven better)
- AdamW + CosineAnnealing (proven better than OneCycleLR)
- SEQ=128 (proven in train_with_test.py)
- Regularization: dropout=0.1, weight_decay=0.1, gradient clipping
- Multi-epoch training (model improves past 1 epoch)
"""
import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, math, sys, os
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ = 128
BATCH = 8
EPOCHS = 5
LR = 5e-4
EVAL_EVERY_STEPS = 2000

train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

class DS(Dataset):
    def __init__(self, toks, seq):
        # Build (x, y) pairs where y is x shifted by 1
        toks_t = torch.tensor(toks, dtype=torch.long)
        n = max(0, (len(toks) - 1) // seq)
        xs, ys = [], []
        for i in range(n):
            start = i * seq
            x_seq = toks_t[start:start + seq]
            y_seq = toks_t[start + 1:start + seq + 1]
            xs.append(x_seq)
            ys.append(y_seq)
        self.x = torch.stack(xs)
        self.y = torch.stack(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = DS(train_toks, SEQ)
val_ds   = DS(val_toks, SEQ)
test_ds  = DS(test_toks, SEQ)

train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_ld   = DataLoader(val_ds, batch_size=BATCH, num_workers=0)
test_ld  = DataLoader(test_ds, batch_size=BATCH, num_workers=0)

print(f"Device: {DEVICE}")
print(f"Train: {len(train_ds):,} samples, Val: {len(val_ds):,}, Test: {len(test_ds):,}")
print(f"Batches: train={len(train_ld)}, val={len(val_ld)}, test={len(test_ld)}")


# === Architecture: Proven winning approach from train_with_test.py ===
class VH:
    """FFT-based circular convolution for position binding."""
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
    """
    HSA Attention: encode → bind positions → accumulate context → denoise → retrieve.
    NO cosine-sim retrieval — simple mean accumulation (proven better).
    """
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1, denoise_iter=3):
        super().__init__()
        self.dm = d_model; self.hd = hsa_dim; self.nh = num_heads
        self.head_dim = hsa_dim // num_heads

        self.encoders = nn.ModuleList([Enc(d_model, self.head_dim) for _ in range(num_heads)])
        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=denoise_iter, binary=False)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hsa_dim, d_model)
        self.drop = nn.Dropout(dropout)

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
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
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids, positions=None):
        B, S = ids.shape
        dev = ids.device
        if positions is None:
            positions = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.drop(self.embed(ids))
        for block in self.blocks:
            x = block(x, positions)
        return self.head(self.norm(x))


def compute_ppl(model, loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    crit = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out.view(-1, 50257), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    return math.exp(total_loss / total_tokens)


def main():
    # Best config from prior runs
    CONFIG = dict(
        vocab_size=50257,
        d_model=320,
        num_layers=3,
        num_heads=4,
        hsa_dim=2048,
        num_experts=4,
        top_k=2,
        d_ff=768,
        router_tt_ranks=[1, 4, 4, 1],
        expert_tt_ranks=[1, 4, 4, 4, 1],
        dropout=0.1,
    )

    model = NSTPModel(**CONFIG).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())

    # Random baseline
    print(f"\n--- Random baseline ---")
    rand_ppl = compute_ppl(model, val_ld)
    print(f"Random val ppl: {rand_ppl:.0f} (expected ~50K)")

    # Training
    total_steps = EPOCHS * len(train_ld)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    crit = nn.CrossEntropyLoss(reduction='mean')

    print(f"\n--- Training: {EPOCHS} epochs, ~{total_steps} steps ---")
    print(f"Params: {params:,} ({params/1e6:.1f}M)")
    print(f"LR: {LR}, Scheduler: CosineAnnealing, WD: 0.1")
    print(f"{'Step':>6}  {'Epoch':>5}  {'Val PPL':>9}  {'Test PPL':>9}  {'Loss':>8}  {'Speed':>7}")
    print("-" * 60)

    t0 = time.time()
    gs = 0
    best_val = float('inf')
    best_state = None
    save_path = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/final_best.pt'
    history = []

    for epoch in range(EPOCHS):
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out.view(-1, 50257), y.view(-1))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gs += 1

            if gs % EVAL_EVERY_STEPS == 0 or gs == total_steps:
                val_ppl  = compute_ppl(model, val_ld)
                test_ppl = compute_ppl(model, test_ld)
                elapsed = time.time() - t0
                speed = gs / elapsed

                epoch_num = gs / len(train_ld)
                mark = " *BEST*" if val_ppl < best_val else ""
                print(f"{gs:>6}  {epoch_num:>5.1f}  {val_ppl:>9.2f}  "
                      f"{test_ppl:>9.2f}  {loss.item():>8.4f}  "
                      f"{speed:.0f} st/s{mark}")

                history.append({
                    'step': gs, 'epoch': epoch_num, 'val_ppl': val_ppl,
                    'test_ppl': test_ppl, 'loss': loss.item()
                })

                if val_ppl < best_val:
                    best_val = val_ppl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(best_state, save_path)

    print(f"\n{'='*60}")
    print(f"Best val: {best_val:.2f}")

    # Full evaluation
    if best_state:
        model.load_state_dict(best_state)

    full_val  = compute_ppl(model, val_ld)
    full_test = compute_ppl(model, test_ld)

    print(f"Full val ppl:   {full_val:.2f}")
    print(f"Full test ppl:  {full_test:.2f}")
    print(f"GPT-2 small:    ~29 (124M params)")
    print(f"vs GPT-2:       {29/full_test:.2f}x")
    print(f"{'='*60}")

    # Save history
    import json
    with open('C:/Users/user/AppData/Local/Temp/nstp-v2/models/training_history.json', 'w') as f:
        json.dump({
            'config': CONFIG,
            'params': params,
            'best_val': best_val,
            'full_val': full_val,
            'full_test': full_test,
            'history': history,
        }, f, indent=2)
    print(f"Saved history + best checkpoint to models/")


if __name__ == '__main__':
    main()