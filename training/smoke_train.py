"""
Minimal smoke training test for NSTP.
"""
import os, sys
sys.path.insert(0, 'C:/Users/online/NSTP')

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from nstp_core import NSTPModel, NSTPConfig, NSTPLoss

device = 'cpu'  # Safe fallback

print(f"Using device: {device}")
print(f"Torch version: {torch.__version__}")

# Small config for fast training
config = NSTPConfig(
    vocab_size=2000,
    d_model=128,
    num_layers=2,
    num_heads=4,
    hsa_dim=2048,
    hsa_bind_mode='xor',
    hsa_binary=True,
    num_experts=4,
    top_k=2,
    d_ff=512,
    router_tt_ranks=[1, 8, 8, 1],
    expert_tt_ranks=[1, 8, 8, 8, 1],
    embedding_tt_ranks=[1, 8, 8, 1],
    use_tt_embedding=True,
)

print("Creating model...")
t0 = time.time()
model = NSTPModel(config).to(device)
print(f"Model created in {time.time()-t0:.2f}s, {model.num_parameters():,} params")

# Loss fn
loss_fn = NSTPLoss(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    hsa_dim=config.hsa_dim,
    num_experts=config.num_experts,
    num_layers=config.num_layers,
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

# Synthetic data: small batch, short seq
B, N = 4, 64

print(f"\nTraining for 50 steps (B={B}, N={N})...")
model.train()
start = time.time()

for step in range(50):
    input_ids = torch.randint(0, config.vocab_size, (B, N), device=device)
    targets = torch.randint(0, config.vocab_size, (B, N), device=device)

    # Forward
    logits, aux_losses = model(input_ids, return_aux_losses=True)

    # Loss
    total_loss, losses = loss_fn(logits, targets, aux_losses)
    loss = total_loss

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0:
        ce = losses.get('ce', torch.tensor(0)).item() if hasattr(losses.get('ce', torch.tensor(0)), 'item') else 0
        bal = losses.get('moe_balance', torch.tensor(0)).item() if hasattr(losses.get('moe_balance', torch.tensor(0)), 'item') else 0
        tt = losses.get('tt_ortho', torch.tensor(0)).item() if hasattr(losses.get('tt_ortho', torch.tensor(0)), 'item') else 0
        print(f"  step {step:3d}: loss={loss.item():.4f}  ce={ce:.4f}  bal={bal:.6f}  tt_ortho={tt:.6f}")

elapsed = time.time() - start
print(f"\nDone in {elapsed:.2f}s ({50/elapsed:.1f} steps/s on {device})")

# Test inference / generation
print("\nGeneration test...")
model.eval()
with torch.no_grad():
    in_ids = torch.randint(0, config.vocab_size, (1, 10))
    out = model.generate(in_ids, max_new_tokens=5, do_sample=False)
    print(f"  Generated: {in_ids.shape} -> {out.shape}")

print("\nSmoke training complete!")
