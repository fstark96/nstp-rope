"""Train RoPE NSTP — 800M tokens."""
import sys, math, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_rope import RoPENSTPModel, CONFIG, DEVICE

print("="*60)
print("RoPE NSTP — Training on 1B tokens (Chinchilla optimal)")
print("="*60)

# Load pre-tokenized data
print("\nLoading pre-tokenized data...")
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_train.npy')
val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_val.npy')
print(f"Train: {len(train_toks)/1e6:.1f}M, Val: {len(val_toks)/1e6:.1f}M")

# Dataset
class DS:
    def __init__(self, toks, seq):
        t = torch.tensor(toks, dtype=torch.long)
        n = max(0, (len(t)-1)//seq)
        self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
        self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

train_ds = DS(train_toks, 128)
val_ds = DS(val_toks, 128)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=32)

print(f"Train batches: {len(train_ld)}")

# Model + optimizer
model = RoPENSTPModel(**CONFIG).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Training
print("\nTraining...")
print(f"{'Step':>6} {'Loss':>8} {'ValPPL':>8} {'Time':>6}")
print("-"*35)

best_ppl = float('inf')
start = time.time()

for step in range(30000):
    model.train()
    for batch_x, batch_y in train_ld:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        with torch.amp.autocast('cuda'):
            logits = model(batch_x)
            loss = criterion(logits.view(-1, 50257), batch_y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        break  # One batch per step
    
    if (step+1) % 1000 == 0:
        model.eval()
        val_loss = 0; val_tokens = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss += criterion(logits.view(-1, 50257), y.view(-1)).item() * x.numel()
                val_tokens += x.numel()
        val_ppl = math.exp(val_loss / val_tokens)
        elapsed = time.time() - start
        
        print(f"{step+1:>6} {loss.item():>8.4f} {val_ppl:>8.2f} {elapsed:>5.0f}s")
        
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save({
                'model': model.state_dict(),
                'val_ppl': val_ppl,
                'step': step+1
            }, 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/rope_nstp.pt')
        
        model.train()

print(f"\nBest Val PPL: {best_ppl:.2f}")

# Final eval with longer contexts
print("\n" + "="*60)
print("Context Length Generalization Test")
print("="*60)
model.eval()
val_t = torch.tensor(val_toks, dtype=torch.long)

for seq in [128, 256, 512, 1024]:
    if len(val_t) < seq + 1:
        continue
    x = val_t[:seq].unsqueeze(0).to(DEVICE)
    y = val_t[1:seq+1].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        ce = criterion(logits.view(-1, 50257), y.view(-1))
    print(f"  SEQ={seq}: Val PPL={math.exp(ce.item()):.2f}")

print("\nWith FFT: SEQ=128→512 gives 70× worse PPL")
print("With RoPE: Should stay stable across lengths!")