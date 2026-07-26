"""Retrain NSTP v1 on 800M tokens (Chinchilla optimal)."""
import sys, os, pickle, math, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from train_final import NSTPModel, DEVICE

CONFIG = dict(vocab_size=50257, d_model=320, num_layers=3, num_heads=4,
              hsa_dim=2048, num_experts=4, top_k=2, d_ff=768,
              router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1], dropout=0.1)

print("="*60)
print("NSTP v1 — Retraining on 800M tokens")
print("Chinchilla optimal: 39M params × 20 tokens/param = 780M tokens")
print("="*60)

# Load 800M tokens
print("\nLoading 800M tokens...")
with open('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_tokens.pkl', 'rb') as f:
    texts = pickle.load(f)
print(f"Loaded {len(texts)} texts")

# Tokenize
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

print("Tokenizing...")
all_tokens = []
for i, text in enumerate(texts):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    all_tokens.extend(tokens)
    if (i+1) % 100000 == 0:
        print(f"  {i+1}/{len(texts)}: {len(all_tokens)/1e6:.1f}M tokens")

all_tokens = np.array(all_tokens, dtype=np.int32)
print(f"Total tokens: {len(all_tokens)/1e6:.1f}M")

# Split into train/val
split = int(0.95 * len(all_tokens))
train_toks = all_tokens[:split]
val_toks = all_tokens[split:]

np.save('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_train.npy', train_toks)
np.save('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_val.npy', val_toks)
print(f"Train: {len(train_toks)/1e6:.1f}M, Val: {len(val_toks)/1e6:.1f}M")

# Create datasets
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

print(f"Train batches: {len(train_ds)//32}")

# Initialize model
model = NSTPModel(**CONFIG).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

# Training loop
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
        
        if (step+1) % 1000 == 0:
            # Evaluate
            model.eval()
            val_loss = 0; val_tokens = 0
            with torch.no_grad():
                for x, y in val_ld:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x)
                    val_loss += criterion(logits.view(-1, 50257), y.view(-1)).item() * x.numel()
                    val_tokens += x.numel()
            val_ppl = math.exp(val_loss / val_tokens)
            
            elapsed = time.time() - start if step == 0 else time.time() - start
            print(f"{step+1:>6} {loss.item():>8.4f} {val_ppl:>8.2f} {elapsed:>5.0f}s")
            
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({'model': model.state_dict(), 'val_ppl': val_ppl, 'step': step+1},
                          'C:/Users/user/AppData/Local/Temp/nstp-v2/models/nstp_800m.pt')
            model.train()
        
        break  # Only do one batch per step for now

print(f"\nBest Val PPL: {best_ppl:.2f}")
