"""
train_scaled.py — Phase 1 training with data scaling
Improvements over train_final.py:
1. SEQ=512 (was 128)
2. Gradient accumulation (effective batch = 1024 tokens)
3. Cosine LR (peak=3e-4, warmup=2pct, min=3e-5)
4. Dropout=0.1
5. AdamW with beta2=0.95, weight_decay=0.1
6. Gradient clipping=1.0
7. AMP mixed precision
8. Target 50K+ steps on FineWeb-Edu 100M tokens
"""
import sys, time, math, os, json, numpy as np, torch, torch.nn as nn

# ── Patch profile ──
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch.nn.functional as F
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE

# ── Config ──
DEVICE      = torch.device('cuda')
SEQ         = 512
VS          = 50257
DM          = 320
NL          = 3
NH          = 4
HSA_DIM     = 2048
NE          = 4
TK          = 2
DFF         = 768
RTR         = [1, 4, 4, 1]
ETR         = [1, 4, 4, 4, 1]
DROPOUT     = 0.1

BATCH       = 2
GRAD_ACCUM  = 128          # effective batch = 2 × 512 × 128 = 131,072 tokens
MAX_STEPS   = 60000
EVAL_EVERY  = 2000
SAVE_EVERY  = 5000
LR_PEAK     = 3e-4
LR_MIN      = 3e-5
WARMUP_PCT  = 0.02
CLIP_GRAD   = 1.0
PATIENCE    = 8
USE_AMP     = True
MODEL_DIR   = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models_scaled'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data ──
DATA_DIR = 'C:/Users/user/AppData/Local/Temp/nstp-v2/data'
train_toks = np.load(f'{DATA_DIR}/fineweb_train_tokens.npy')
val_toks   = np.load(f'{DATA_DIR}/fineweb_val_tokens.npy')
test_toks  = np.load(f'{DATA_DIR}/fineweb_test_tokens.npy')
print(f"Train={len(train_toks):,} Val={len(val_toks):,} Test={len(test_toks):,}")

class LMDataset:
    def __init__(self, toks, seq):
        t = torch.tensor(toks, dtype=torch.long)
        n = max(0, (len(t) - 1) // seq)
        self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
        self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

train_ds = LMDataset(train_toks, SEQ)
val_ds   = LMDataset(val_toks, SEQ)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                       num_workers=0, pin_memory=True, drop_last=True)
val_ld   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, num_workers=0)
BATCHES  = len(train_ds) // BATCH
print(f"Train batches/epoch: {BATCHES}, Val examples: {len(val_ds)}")

# ── Model (identical to train_final.py) ──
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
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5); nn.init.zeros_(self.proj.bias)
    def forward(self, x): return F.normalize(self.proj(x), p=2, dim=-1)

class HDAAttn(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1, denoise_iter=3):
        super().__init__()
        self.dm = d_model; self.hd = hsa_dim; self.nh = num_heads
        self.head_dim = hsa_dim // num_heads
        self.encoders = nn.ModuleList([Enc(d_model, self.head_dim) for _ in range(num_heads)])
        self.denoisers = nn.ModuleList([HSADenoiser(self.head_dim, num_iterations=denoise_iter, binary=False) for _ in range(num_heads)])
        self.out_proj = nn.Linear(hsa_dim, d_model); self.drop = nn.Dropout(dropout)
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
    def __init__(self, d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                 router_tt_ranks, expert_tt_ranks, dropout=0.1):
        super().__init__()
        self.attn = HDAAttn(d_model, hsa_dim, num_heads, dropout)
        self.moe = TTCERMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
                            router_tt_ranks=router_tt_ranks, expert_tt_ranks=expert_tt_ranks,
                            activation='gelu', dropout=dropout, router_aux_loss_coef=0.01)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, positions):
        r = x; x = self.norm1(x); a, _ = self.attn(x, positions); x = r + self.drop(a)
        r = x; x = self.norm2(x); m, _ = self.moe(x); return r + m

class NSTPModel(nn.Module):
    def __init__(self, vs, dm, nl, nh, hd, ne, tk, dff, rtr, etr, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vs, dm)
        self.blocks = nn.ModuleList([NSTPBlock(dm, hd, nh, ne, tk, dff, rtr, etr, dropout) for _ in range(nl)])
        self.norm = nn.LayerNorm(dm); self.head = nn.Linear(dm, vs, bias=False)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, ids, positions=None):
        B, S = ids.shape; dev = ids.device
        if positions is None: positions = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.drop(self.embed(ids))
        for b in self.blocks: x = b(x, positions)
        return self.head(self.norm(x))

print(f"\nBuilding model: {DM}d × {NL}L × {NH}H × {HSA_DIM}hd × {NE}E")
model = NSTPModel(VS, DM, NL, NH, HSA_DIM, NE, TK, DFF, RTR, ETR, DROPOUT).to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

# ── Optimizer: AdamW with beta2=0.95 ──
opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK, betas=(0.9, 0.95),
                        weight_decay=0.1, eps=1e-8)

# ── Cosine LR schedule ──
warmup_steps = int(WARMUP_PCT * MAX_STEPS)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR_PEAK,
    total_steps=MAX_STEPS,
    pct_start=WARMUP_PCT,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=LR_PEAK / LR_MIN
)

# ── Mixed precision ──
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

crit = nn.CrossEntropyLoss(reduction='mean')
best_val = float('inf')
patience_ctr = 0
train_losses = []
val_ppls = []

# ── Load checkpoint if exists ──
CKPT = f'{MODEL_DIR}/scaled_best.pt'
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model'])
    best_val = ckpt['val_ppl']
    start_step = ckpt.get('step', 0)
    print(f"Loaded checkpoint: step={start_step}, val_ppl={best_val:.2f}")
else:
    start_step = 0
    print("Fresh start")

print(f"\nStarting training: {MAX_STEPS} steps, LR={LR_PEAK}, warmup={warmup_steps}, grad_accum={GRAD_ACCUM}")
print(f"Effective batch: {BATCH} × {SEQ} × {GRAD_ACCUM} = {BATCH*SEQ*GRAD_ACCUM:,} tokens")
print(f"{'Step':>6} {'Train CE':>10} {'Val PPL':>10} {'Time':>6} {'Note'}")
print("-" * 60)

step = 0
epoch = 0
opt.zero_grad()
start_time = time.time()

while step < MAX_STEPS:
    epoch += 1
    for x_batch, y_batch in train_ld:
        if step >= MAX_STEPS: break
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            out = model(x_batch)
            loss = crit(out.view(-1, VS), y_batch.view(-1))
        
        loss_scaled = loss / GRAD_ACCUM
        scaler.scale(loss_scaled).backward()
        
        train_losses.append(loss.item())
        
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            scheduler.step()
        
        step += 1
        
        # ── Eval ──
        if step % EVAL_EVERY == 0:
            model.eval()
            total_loss, total_tokens = 0.0, 0
            with torch.no_grad():
                for vx, vy in val_ld:
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    out = model(vx)
                    l = crit(out.view(-1, VS), vy.view(-1))
                    total_loss += l.item() * vx.numel()
                    total_tokens += vx.numel()
            val_ppl = math.exp(total_loss / total_tokens)
            elapsed = time.time() - start_time
            train_ce = np.mean(train_losses[-EVAL_EVERY*10:]) if train_losses else 0
            
            note = ""
            if val_ppl < best_val:
                best_val = val_ppl
                patience_ctr = 0
                torch.save({'model': model.state_dict(), 'val_ppl': val_ppl, 'step': step,
                           'params': params}, CKPT)
                note = f"*BEST* → {CKPT}"
            else:
                patience_ctr += 1
            
            print(f"{step:>6} {train_ce:>10.4f} {val_ppl:>10.2f} {elapsed:>5.0f}s {note}")
            model.train()
            
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at step {step} (patience={PATIENCE})")
                break
    
    # Shuffle each epoch
    if step < MAX_STEPS:
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=True)

# ── Final test ──
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True)['model'])
model.eval()
total_loss, total_tokens = 0.0, 0
with torch.no_grad():
    for x, y in torch.utils.data.DataLoader(
        LMDataset(test_toks, SEQ), batch_size=BATCH, num_workers=0):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        l = crit(out.view(-1, VS), y.view(-1))
        total_loss += l.item() * x.numel()
        total_tokens += x.numel()
test_ppl = math.exp(total_loss / total_tokens)
print(f"\n{'='*60}")
print(f"FINAL: Best val={best_val:.2f}  Test={test_ppl:.2f}  Step={step}")
print(f"{'='*60}")
