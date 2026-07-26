"""Test NSTP v1 on longer contexts — SEQ=512 and SEQ=1024."""
import sys, os, math, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from train_final import NSTPModel, DEVICE

CONFIG = dict(vocab_size=50257, d_model=320, num_layers=3, num_heads=4,
              hsa_dim=2048, num_experts=4, top_k=2, d_ff=768,
              router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1], dropout=0.1)

print("Loading NSTP v1 model...")
model = NSTPModel(**CONFIG).to(DEVICE)
ckpt = torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models/final_best.pt', 
                   map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f"Model loaded (Val PPL=3.82 at SEQ=128)")

val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

for SEQ in [128, 256, 512, 1024]:
    print(f"\n{'='*60}")
    print(f"Testing SEQ={SEQ}")
    print(f"{'='*60}")
    
    # Create datasets
    class DS:
        def __init__(self, toks, seq):
            t = torch.tensor(toks, dtype=torch.long)
            n = max(0, (len(t)-1)//seq)
            self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
            self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
        def __len__(self): return len(self.xs)
        def __getitem__(self, i): return self.xs[i], self.ys[i]
    
    val_ds = DS(val_toks, SEQ)
    test_ds = DS(test_toks, SEQ)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=4)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=4)
    
    crit = nn.CrossEntropyLoss(reduction='mean')
    
    # Compute val PPL
    tl, tt = 0.0, 0
    start = time.time()
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            lo = model(x)
            tl += crit(lo.view(-1, 50257), y.view(-1)).item() * x.numel()
            tt += x.numel()
    val_ppl = math.exp(tl / tt)
    
    # Compute test PPL
    tl, tt = 0.0, 0
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            lo = model(x)
            tl += crit(lo.view(-1, 50257), y.view(-1)).item() * x.numel()
            tt += x.numel()
    test_ppl = math.exp(tl / tt)
    
    elapsed = time.time() - start
    print(f"  Val PPL:  {val_ppl:.2f}")
    print(f"  Test PPL: {test_ppl:.2f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Val examples: {len(val_ds)}, Test examples: {len(test_ds)}")

print(f"\n{'='*60}")
print("CONTEXT LENGTH COMPARISON")
print(f"{'='*60}")
print("SEQ=128 is the training context length.")
print("Longer contexts should give LOWER PPL (more context = easier prediction).")
print("If PPL increases with context length, the model isn't using the extra context.")
