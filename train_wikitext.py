import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, math
sys.path.insert(0, '/tmp/nstp-v2')
from nstp_core.model import NSTPModel, NSTPConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, SEQ = 4, 128
LR = 3e-4
EPOCHS = 3
EVAL_EVERY = 500

config = NSTPConfig(
    vocab_size=50257, d_model=256, num_layers=4, num_heads=4,
    hsa_dim=2048, hsa_binary=True, num_experts=4, top_k=2, d_ff=512,
    router_tt_ranks=[1,8,8,1], expert_tt_ranks=[1,8,8,8,1], dropout=0.1)

model = NSTPModel(config).to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params:,} params  ({params/1e6:.1f}M)")
print(f"Device: {DEVICE}")

train_toks = np.load('data/wikitext2_train_tokens.npy')
val_toks   = np.load('data/wikitext2_validation_tokens.npy')
test_toks  = np.load('data/wikitext2_test_tokens.npy')

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
test_ds  = DS(test_toks,  SEQ)

train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH, num_workers=0)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(train_ld))
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

print(f"\nTraining for {EPOCHS} epochs...")
print(f"Train: {len(train_ds)} batches | Val: {len(val_ds)} batches | Seq: {SEQ}")
print(f"{'Step':>6} {'Loss':>8} {'TrainPPL':>9} {'ValPPL':>8} {'LR':>10} {'Etc':>6}")
print("-"*55)

t0 = time.time()
gs, best = 0, float('inf')

for ep in range(EPOCHS):
    model.train()
    ep_loss, ep_tok = 0, 0
    for bi, (x,y) in enumerate(train_ld):
        x,y = x.to(DEVICE), y.to(DEVICE)
        out,_ = model(x)
        if isinstance(out, tuple): out=out[0]
        loss = crit(out.view(-1,config.vocab_size), y.view(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        ep_loss += loss.item()*x.numel(); ep_tok += x.numel(); gs += 1
        if gs % EVAL_EVERY == 0:
            vp = ppl(val_ld); tp = math.exp(ep_loss/ep_tok); lr = sched.get_last_lr()[0]
            etc = time.time()-t0
            print(f"{gs:>6} {loss.item():.4f} {tp:>9.2f} {vp:>8.2f} {lr:.2e} {etc:.0f}s")
            if vp < best:
                best = vp
                torch.save(model.state_dict(), 'best_nstp.pt')
                print(f"  -> Saved best (val_ppl={vp:.2f})")

    vp = ppl(val_ld)
    tp = math.exp(ep_loss/ep_tok)
    print(f"\nEpoch {ep+1}: train_ppl={tp:.2f}  val_ppl={vp:.2f}")

model.load_state_dict(torch.load('best_nstp.pt'))
tppl = ppl(torch.utils.data.DataLoader(test_ds, batch_size=BATCH))
print(f"\n=== FINAL RESULTS ===")
print(f"Test perplexity: {tppl:.2f}")
print(f"GPT-2 small:     ~29.41")
print(f"Ratio:           {tppl/29.41:.1f}x")