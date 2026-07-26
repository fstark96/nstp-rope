import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, numpy as np, time, math
sys.path.insert(0, '/tmp/nstp-v2')
from nstp_core.model import NSTPModel, NSTPConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, SEQ = 4, 128
LR = 3e-4
STEPS = 2000
EVAL_EVERY = 500

def make_config(trainable=True, hsa_dim=1024, num_layers=2, num_experts=2, d_model=128, d_ff=256):
    return NSTPConfig(
        vocab_size=50257, d_model=d_model, num_layers=num_layers, num_heads=4,
        hsa_dim=hsa_dim, hsa_binary=True, num_experts=num_experts, top_k=2, d_ff=d_ff,
        router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1], dropout=0.1,
        hsa_trainable_encoder=trainable, use_tt_embedding=False)

def train_model(config, name):
    model = NSTPModel(config).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())

    train_toks = np.load('data/wikitext2_train_tokens.npy')
    val_toks   = np.load('data/wikitext2_validation_tokens.npy')

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

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    crit = nn.CrossEntropyLoss()

    def ppl(loader):
        model.eval()
        loss, tok = 0, 0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                out,_ = model(x)
                if isinstance(out, tuple): out=out[0]
                loss += crit(out.view(-1,config.vocab_size), y.view(-1)).item() * x.numel()
                tok  += x.numel()
        return math.exp(loss/tok)

    print(f"\n{'='*60}")
    print(f"{name}: {params:,} params  ({params/1e6:.1f}M)")
    print(f"{'='*60}")

    t0 = time.time()
    gs = 0
    best_val = float('inf')

    for x,y in train_ld:
        if gs >= STEPS: break
        x,y = x.to(DEVICE), y.to(DEVICE)
        out,_ = model(x)
        if isinstance(out, tuple): out=out[0]
        loss = crit(out.view(-1,config.vocab_size), y.view(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        gs += 1
        if gs % EVAL_EVERY == 0:
            vp = ppl(val_ld)
            elapsed = time.time()-t0
            print(f"  Step {gs:5d}: val_ppl={vp:.1f}  ({elapsed:.0f}s)")
            if vp < best_val: best_val = vp

    print(f"  Final (step {gs}): best_val_ppl={best_val:.1f}")
    return best_val

# Run both
val_fixed = train_model(make_config(trainable=False), "FIXED encoder (original)")
val_train = train_model(make_config(trainable=True), "TRAINABLE encoder (THDC-style)")
val_train_big = train_model(make_config(trainable=True, hsa_dim=2048, d_model=256, d_ff=512), "TRAINABLE 2K-dim")

print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"  FIXED encoder:        val_ppl = {val_fixed:.1f}")
print(f"  TRAINABLE encoder:    val_ppl = {val_train:.1f}  ({val_fixed/val_train:.2f}x vs fixed)")
print(f"  TRAINABLE 2K-dim:     val_ppl = {val_train_big:.1f}  ({val_fixed/val_train_big:.2f}x vs fixed)")
print(f"  GPT-2 small baseline:  ~29.4")