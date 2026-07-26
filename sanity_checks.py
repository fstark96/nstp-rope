"""Sanity checks for NSTP eval — per ChatGPT's recommendations"""
import sys
sys.modules['profile'] = type(sys)('fp')

import torch, math, numpy as np, torch.nn as nn, torch.nn.functional as F, sys
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE

DEVICE = torch.device('cuda')
SEQ, VS = 128, 50257

# Load checkpoint (same model from train_final.py)
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
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
    def forward(self, x, positions):
        r = x; x = self.norm1(x); a, _ = self.attn(x, positions); x = r + self.drop(a)
        r = x; x = self.norm2(x); m, _ = self.moe(x); return r + m

class NSTPModel(nn.Module):
    def __init__(self, vs, dm, nl, nh, hd, ne, tk, dff, rtr, etr, drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(vs, dm)
        self.blocks = nn.ModuleList([NSTPBlock(dm, hd, nh, ne, tk, dff, rtr, etr, drop) for _ in range(nl)])
        self.norm = nn.LayerNorm(dm); self.head = nn.Linear(dm, vs, bias=False); self.drop = nn.Dropout(drop)
        self.apply(lambda m: (nn.init.normal_(m.weight, 0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else
              (nn.init.ones_(m.weight), nn.init.zeros_(m.bias)) if isinstance(m, nn.LayerNorm) else None))
    def forward(self, ids, positions=None):
        B, S = ids.shape; dev = ids.device
        if positions is None: positions = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.drop(self.embed(ids))
        for b in self.blocks: x = b(x, positions)
        return self.head(self.norm(x))

model = NSTPModel(50257, 320, 3, 4, 2048, 4, 2, 768, [1,4,4,1], [1,4,4,4,1], 0.1).to(DEVICE)
sd = torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models/final_best.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(sd)
model.eval()

# Load data
val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
class DS:
    def __init__(self, toks, seq):
        t = torch.tensor(toks, dtype=torch.long)
        n = max(0, (len(t)-1)//seq)
        self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
        self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

val_ds = DS(val_toks, SEQ)
print(f"Val size: {len(val_ds)} examples, {SEQ} tokens each")
print(f"Val token total: {len(val_ds) * SEQ:,}")

# =============================================================
# CHECK 1: Are labels shifted correctly? Print first example
# =============================================================
print("\n" + "="*60)
print("CHECK 1: Label shift verification")
print("="*60)
# Get first example from dataset
x0, y0 = val_ds[0]  # both are [128]
print(f"x[0:20]: {x0[:20].tolist()}")
print(f"y[0:20]: {y0[:20].tolist()}")
print(f"y[i] == x[i+1] for i=0..10: {[y0[i].item() == x0[i+1].item() for i in range(10)]}")
print(f"Labels are shifted by 1: {all(y0[i].item() == x0[i+1].item() for i in range(127))}")

# =============================================================
# CHECK 2: Does model predict current token instead of next?
# =============================================================
print("\n" + "="*60)
print("CHECK 2: Prediction sanity — does model predict current or next token?")
print("="*60)
x_batch = val_ds.xs[:5].to(DEVICE)  # [5, 128]
y_batch = val_ds.ys[:5].to(DEVICE)  # [5, 128]
pos = torch.arange(SEQ, device=DEVICE).unsqueeze(0).expand(5, -1)

with torch.no_grad():
    logits = model(x_batch, pos)  # [5, 128, 50257]
    preds = logits.argmax(dim=-1)  # [5, 128]
    probs = F.softmax(logits, dim=-1)
    
    # Check: does model predict CURRENT input token (BUG) or NEXT target?
    correct_current = (preds == x_batch).float().mean().item()  # predicting input
    correct_target = (preds == y_batch).float().mean().item()  # predicting target
    
    print(f"Accuracy vs CURRENT token (input):  {correct_current:.4f} ({correct_current*100:.1f}%)")
    print(f"Accuracy vs NEXT token (target):    {correct_target:.4f} ({correct_target*100:.1f}%)")
    
    if correct_current > 0.5:
        print("🚨 BUG: Model is predicting current token — NOT the next token!")
    elif correct_target > 0.5:
        print("✅ Model is correctly predicting next token (target)")
    else:
        print("⚠️ Model is predicting something else")

    # Print top-3 predictions for first 5 positions
    print("\nFirst 5 positions — input / target / top3_preds / probs:")
    for i in range(5):
        inp = x_batch[0, i].item()
        tgt = y_batch[0, i].item()
        top3 = preds[0, i].item()
        p3 = probs[0, i, top3].item()
        print(f"  pos={i}: input={inp} target={tgt} top_pred={top3} p={p3:.4f} correct={'✅' if top3==tgt else '❌'}")

# =============================================================
# CHECK 3: Validate random-permuted labels give high PPL
# =============================================================
print("\n" + "="*60)
print("CHECK 3: Random permutation test")
print("="*60)
y_random = y_batch.clone()
for i in range(y_random.numel()):
    y_random.view(-1)[i] = torch.randint(0, VS, (1,)).item()

with torch.no_grad():
    logits = model(x_batch, pos)
    crit = nn.CrossEntropyLoss(reduction='mean')
    loss_real = crit(logits.view(-1, VS), y_batch.view(-1))
    loss_rand = crit(logits.view(-1, VS), y_random.view(-1))
    print(f"CE with real labels:    {loss_real.item():.4f} → PPL = {math.exp(loss_real.item()):.2f}")
    print(f"CE with random labels:  {loss_rand.item():.4f} → PPL = {math.exp(loss_rand.item()):.2f}")
    if math.exp(loss_rand.item()) < 100:
        print("⚠️ Random labels still give low PPL — suggests label leakage!")
    else:
        print("✅ Random labels give high PPL — labels are being used correctly")

# =============================================================
# CHECK 4: Verify train/val/test splits are different
# =============================================================
print("\n" + "="*60)
print("CHECK 4: Data split verification")
print("="*60)
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
test_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')
val_toks_full = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
print(f"Train tokens: {len(train_toks):,}")
print(f"Val tokens:   {len(val_toks_full):,}")
print(f"Test tokens:  {len(test_toks):,}")
# Check for overlap: first 100 train vs first 100 val
overlap_train_val = len(set(train_toks[:1000]) & set(val_toks_full[:1000]))
overlap_val_test = len(set(val_toks_full[:1000]) & set(test_toks[:1000]))
print(f"First 1000-token overlap: train-val={overlap_train_val}, val-test={overlap_val_test}")
print(f"Data splits are {'segregated ✅' if overlap_train_val < 100 and overlap_val_test < 100 else '⚠️ OVERLAPPING'}")

# =============================================================
# CHECK 5: Causal mask is active — VH.bind/unbind are correct
# =============================================================
print("\n" + "="*60)
print("CHECK 5: VH bind/unbind — does information flow correctly?")
print("="*60)
# Test: can information from position i reach position i+1?
# In VH.bind: h_bound = FFT(h * rot(pos)), all positions contribute to all freq bins
# In VH.unbind: each position gets back all frequency components
# But: M = h_bound.mean(dim=1) — the mean is taken across positions
# This means: M encodes information from ALL positions (global average)
# Then VH.unbind(M, ...) returns the same M-bound signal to ALL positions
# So position i can see information from ALL other positions through M!
# 
# Wait, let me verify: does VH.unbind produce the same tensor for all positions?
test_h = torch.randn(2, 128, 320, device=DEVICE)
test_pos = torch.arange(128, device=DEVICE).unsqueeze(0).expand(2, -1)
with torch.no_grad():
    h_bound = VH.bind(test_h, test_pos, 512)  # each position bound
    M = h_bound.mean(dim=1)  # [B, 512], global mean
    h_unbound = VH.unbind(M, test_pos, 512)   # [B, 128, 512]
    # Is the unbound signal the same for all positions?
    variance_across_pos = h_unbound.std(dim=1).mean().item()
    print(f"Std of h_ret across positions (should be 0 if identical): {variance_across_pos:.6f}")
    if variance_across_pos < 1e-5:
        print("🚨 All positions get IDENTICAL h_ret — no positional differentiation!")
        print("   Model cannot distinguish position 0 from position 127!")
        print("   This means the model is predicting from the MEAN embedding, not from position-specific info")
    else:
        print(f"h_ret varies across positions: {variance_across_pos:.4f}")

# =============================================================
# CHECK 6: PPL computation formula
# =============================================================
print("\n" + "="*60)
print("CHECK 6: PPL formula verification")
print("="*60)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=8, num_workers=0)
crit = nn.CrossEntropyLoss(reduction='mean')
total_loss, total_tokens = 0.0, 0
with torch.no_grad():
    for x, y in val_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)  # positions=None
        loss = crit(out.view(-1, VS), y.view(-1))
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
ppl = math.exp(total_loss / total_tokens)
print(f"Total tokens: {total_tokens:,}")
print(f"Sum NLL: {total_loss:.4f}")
print(f"Mean NLL: {total_loss/total_tokens:.4f}")
print(f"PPL = exp(mean_NLL) = {ppl:.4f}")
print(f"Formula: PPL = exp({total_loss/total_tokens:.4f}) = {ppl:.4f}")
print("✅ Formula matches ChatGPT's expected: exp(mean NLL)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)