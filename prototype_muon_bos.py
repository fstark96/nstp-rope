"""
prototype_muon_bos.py — Fast prototype: Muon + BOS-packing only.
Tests the two most important V3 additions on real FineWeb data for ~500 steps.
Goal: verify (a) Muon doesn't crash on NSTP-Ω params, (b) BOS-packing works on real data.
"""
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
import sys
sys.modules['profile'] = FakeProfile()

sys.stdout.reconfigure(line_buffering=True)

import os
import time
import math
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_omega_v3 import NSTPOmegaV3, NSTPOmegaConfig, split_v3_params_for_muon
from muon_optimizer import Muon
from bos_packing import BOSPackedDataset, BOSPackedStreamingLoader
from val_bpb import compute_val_bpb

import torch
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda')
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

print("=" * 70)
print("PROTOTYPE: MUON + BOS-PACKING ON FINEWEB (500 STEPS)")
print("=" * 70)

# Data
fw_train = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_train.npy', mmap_mode='r')
fw_val = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_val.npy', mmap_mode='r')
print(f"Train: {len(fw_train):,} | Val: {len(fw_val):,}")

# Config — small for fast prototype
SEQ_LEN = 256
BATCH_SIZE = 4
ACCUM_STEPS = 8
LR_MUON = 0.001   # Safe Muon LR for NSTP-Ω scale (nanochat uses 0.02 but model is bigger)
LR_ADAMW = 0.001
MAX_STEPS = 100
LOG_EVERY = 10
EVAL_STEPS = 100

config = NSTPOmegaConfig(
    vocab_size=50257, d_model=512, num_layers=6, num_heads=8,
    num_experts=4, head_dim=64, hhm_l2_dim=2048, hhm_l3_dim=8192,
    hhm_num_prototypes=512, tt_ranks=[2, 4, 8, 2], dropout=0.1,
    layer_drop=0.0, min_layers=3, max_layers=6, halt_threshold=1.0,
    target_sparsity=1.0,
)

model = NSTPOmegaV3(config).to(DEVICE)
model.hhm = None  # Disable HHM for prototype
total = sum(p.numel() for p in model.parameters())
print(f"Model: {total:,} params ({total/1e6:.1f}M)")

# Split params
muon_param_list, adamw_param_list = split_v3_params_for_muon(model)
muon_params = [p for _, p in muon_param_list]
adamw_params = [p for _, p in adamw_param_list]

muon_opt = Muon(muon_params, lr=LR_MUON, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.0)
adamw_opt = torch.optim.AdamW(adamw_params, lr=LR_ADAMW, betas=(0.8, 0.95), eps=1e-10)
print(f"Muon: {len(muon_params)} tensors ({sum(p.numel() for p in muon_params):,})")
print(f"AdamW: {len(adamw_params)} tensors ({sum(p.numel() for p in adamw_params):,})")

# BOS-packing data
print("\nSetting up BOS-packed data...")
train_loader = BOSPackedStreamingLoader(
    fw_train[:5_000_000],  # 5M tokens for prototype
    seq_len=SEQ_LEN, batch_size=BATCH_SIZE, buffer_size=50
)
val_subset = fw_val[:200_000].copy()
for i in range(0, len(val_subset), 256):
    val_subset[i] = 50256
val_ds = BOSPackedDataset(val_subset, seq_len=SEQ_LEN, buffer_size=50)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
print(f"Train loader: streaming (5M tokens), Val: {len(val_ds):,} rows")

# Training loop
print("\n" + "=" * 70)
print(f"STARTING {MAX_STEPS} STEPS")
print("=" * 70)

train_iter = iter(train_loader)
global_step = 0
tokens_seen = 0
start = time.time()
losses = []

try:
    while global_step < MAX_STEPS:
        model.reset_memory()
        accum_loss = 0.0
        for micro in range(ACCUM_STEPS):
            x, y = next(train_iter)
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x, return_drafts=False, softcap=15.0)
                logits = out['logits']
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            (loss / ACCUM_STEPS).backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step()
        adamw_opt.step()
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        global_step += 1
        losses.append(accum_loss / ACCUM_STEPS)
        avg_loss = accum_loss / ACCUM_STEPS

        if global_step % LOG_EVERY == 0:
            elapsed = time.time() - start
            tok_per_sec = tokens_seen / elapsed
            recent = losses[-LOG_EVERY:]
            avg_loss_recent = sum(recent) / len(recent)
            print(f"  step {global_step:4d}/{MAX_STEPS} | loss {avg_loss_recent:.4f} | "
                  f"tok/s {tok_per_sec:,.0f} | time {elapsed:.0f}s")

        if math.isnan(avg_loss) or avg_loss > 100:
            print(f"  FAIL: loss exploded at step {global_step} (avg_loss={avg_loss:.4f})")
            break

    # Eval
    print("\nFinal evaluation...")
    model.eval()
    model.reset_memory()
    val_bpb = compute_val_bpb(model, val_loader, DEVICE, vocab_size=50257, max_eval_batches=20)
    print(f"Final val_bpb: {val_bpb:.4f}")
    print(f"Initial loss:  {losses[0]:.4f}")
    print(f"Final loss:    {losses[-1]:.4f}")
    print(f"Reduction:     {(1 - losses[-1]/losses[0])*100:.1f}%")
    print(f"Total time:    {time.time()-start:.0f}s")
    print(f"Tokens:        {tokens_seen:,}")

    print("\n✅ PROTOTYPE PASSED — Muon + BOS-packing work on real data")

except Exception as e:
    print(f"\n❌ PROTOTYPE FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
