"""
train_nstp_omega_v3.py — NSTP-Ω V3 Training

Integrates all Karpathy/autoresearch improvements:
- Muon optimizer for 2D matrices + AdamW for embeddings/scalars
- BOS-aligned best-fit packing (no padding)
- val_bpb evaluation (vocab-independent metric)
- Value Embeddings, x0 lambdas, softcap logits (built into model)
- Fast-fail on NaN/loss>100
- GC freeze after warmup
- Warmup + steady + warmdown LR schedule

V2 training is still running untouched (separate files).
This script writes to models_v3/ — never touches models_v2/.
"""
# Fix profile module conflict BEFORE any other imports
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
import sys
sys.modules['profile'] = FakeProfile()

import os
import gc
import time
import math
import json
import logging
from datetime import datetime
from pathlib import Path

# Optional experiment config override (used by run_experiment.py / autoresearch_loop.py)
_EXP_CONFIG = {}
_exp_config_path = os.environ.get('EXPERIMENT_CONFIG')
if _exp_config_path and os.path.exists(_exp_config_path):
    try:
        _EXP_CONFIG = json.loads(Path(_exp_config_path).read_text())
        print(f"Loaded experiment config from {_exp_config_path}: {json.dumps(_EXP_CONFIG, indent=2)}")
    except Exception as e:
        print(f"Failed to load experiment config: {e}")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

STATUS_FILE = Path('C:/Users/user/AppData/Local/Temp/nstp-v2/logs/training_v3_status.txt')

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            with open(STATUS_FILE, 'a') as f:
                f.write(self.format(record) + '\n')
        except:
            pass

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

log_dir = Path('C:/Users/user/AppData/Local/Temp/nstp-v2/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'training_v3_{datetime.now():%Y%m%d_%H%M%S}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[FlushFileHandler(log_file), FlushStreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("NSTP-Ω V3 TRAINING (MUON + BOS-PACKING + VAL_BPB + VE)")
logger.info("=" * 70)

sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_omega_v3 import NSTPOmegaV3, NSTPOmegaConfig, split_v3_params_for_muon
from muon_optimizer import Muon, split_params_for_muon
from bos_packing import BOSPackedDataset, BOSPackedStreamingLoader
from val_bpb import compute_val_bpb, compute_val_ppl, estimate_token_bytes

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# DATA
# ============================================================================
logger.info("Loading FineWeb-Edu data...")
fw_train = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_train.npy', mmap_mode='r')
fw_val = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_800m_val.npy', mmap_mode='r')
logger.info(f"Train: {len(fw_train):,} tokens | Val: {len(fw_val):,} tokens")

# ============================================================================
# CONFIG
# ============================================================================
SEQ_LEN = _EXP_CONFIG.get('SEQ_LEN', 512)
BATCH_SIZE = _EXP_CONFIG.get('BATCH_SIZE', 4)
ACCUM_STEPS = _EXP_CONFIG.get('ACCUM_STEPS', 16)
LR_MUON = _EXP_CONFIG.get('LR_MUON', 0.02)          # Karpathy's default for matrix LR
LR_ADAMW_EMBED = _EXP_CONFIG.get('LR_ADAMW_EMBED', 0.6)    # embeddings get higher LR
LR_ADAMW_HEAD = _EXP_CONFIG.get('LR_ADAMW_HEAD', 0.004)   # lm_head gets lower LR
LR_SCALAR = _EXP_CONFIG.get('LR_SCALAR', 0.5)         # per-layer scalars
WEIGHT_DECAY = _EXP_CONFIG.get('WEIGHT_DECAY', 0.0)
GRAD_CLIP = _EXP_CONFIG.get('GRAD_CLIP', 1.0)
EVAL_EVERY = _EXP_CONFIG.get('EVAL_EVERY', 2500)
SAVE_EVERY = _EXP_CONFIG.get('SAVE_EVERY', 5000)
MAX_STEPS = _EXP_CONFIG.get('MAX_STEPS', 50000)
LOG_EVERY = _EXP_CONFIG.get('LOG_EVERY', 100)
WARMUP_RATIO = _EXP_CONFIG.get('WARMUP_RATIO', 0.01)     # 500 steps warmup
WARMDOWN_RATIO = _EXP_CONFIG.get('WARMDOWN_RATIO', 0.5)    # 50% warmdown
FINAL_LR_FRAC = _EXP_CONFIG.get('FINAL_LR_FRAC', 0.0)
SOFTCAP = _EXP_CONFIG.get('SOFTCAP', 15.0)
NUM_WORKERS = 0

SAVE_DIR = Path(_EXP_CONFIG.get('_SAVE_DIR', 'C:/Users/user/AppData/Local/Temp/nstp-v2/models_v3'))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = SAVE_DIR / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL
# ============================================================================
config = NSTPOmegaConfig(
    vocab_size=50257,
    d_model=512,
    num_layers=6,
    num_heads=8,
    num_experts=4,
    head_dim=64,
    hhm_l2_dim=2048,
    hhm_l3_dim=8192,
    hhm_num_prototypes=512,
    tt_ranks=[2, 4, 8, 2],
    dropout=0.1,
    layer_drop=0.0,
    min_layers=3,
    max_layers=6,
    halt_threshold=1.0,
    target_sparsity=1.0,
)

model = NSTPOmegaV3(config).to(DEVICE)
model.hhm = None  # Disable HHM for now (it causes memory issues with gradient flow)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"NSTP-Ω V3: {total_params:,} params ({total_params/1e6:.1f}M)")

# ============================================================================
# SPLIT PARAMS: MUON vs ADAMW
# ============================================================================
muon_param_list, adamw_param_list = split_v3_params_for_muon(model)
logger.info(f"Muon params:    {len(muon_param_list)} tensors, {sum(p.numel() for _, p in muon_param_list):,} params")
logger.info(f"AdamW params:   {len(adamw_param_list)} tensors, {sum(p.numel() for _, p in adamw_param_list):,} params")

# Build optimizer groups
muon_params_only = [p for _, p in muon_param_list]
adamw_params_only = [p for _, p in adamw_param_list]

muon_opt = Muon(muon_params_only, lr=LR_MUON, momentum=0.95,
                ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY, cautious_wd=True)

# Separate AdamW groups for embeddings vs head vs scalars
embed_params, head_params, scalar_params = [], [], []
for name, p in adamw_param_list:
    if 'embed' in name or 'wte' in name:
        embed_params.append(p)
    elif '.head.' in name or 'lm_head' in name:
        head_params.append(p)
    else:
        scalar_params.append(p)

adamw_opt = torch.optim.AdamW([
    {'params': embed_params, 'lr': LR_ADAMW_EMBED, 'betas': (0.8, 0.95), 'eps': 1e-10, 'weight_decay': 0.0},
    {'params': head_params, 'lr': LR_ADAMW_HEAD, 'betas': (0.8, 0.95), 'eps': 1e-10, 'weight_decay': 0.0},
    {'params': scalar_params, 'lr': LR_SCALAR, 'betas': (0.96, 0.95), 'eps': 1e-10, 'weight_decay': 0.0},
], lr=LR_ADAMW_EMBED)

logger.info(f"  Embedding AdamW: {len(embed_params)} tensors, lr={LR_ADAMW_EMBED}")
logger.info(f"  Head AdamW:      {len(head_params)} tensors, lr={LR_ADAMW_HEAD}")
logger.info(f"  Scalar AdamW:    {len(scalar_params)} tensors, lr={LR_SCALAR}")

# ============================================================================
# DATA LOADERS (BOS-aligned best-fit packing)
# ============================================================================
logger.info("Building BOS-aligned packed dataset...")
val_ds = BOSPackedDataset(fw_val, seq_len=SEQ_LEN, buffer_size=50)

# Streaming for train (re-shuffles per epoch)
train_loader = BOSPackedStreamingLoader(
    fw_train, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, buffer_size=100
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

logger.info(f"Val rows: {len(val_ds):,}")

# ============================================================================
# LR SCHEDULE: WARMUP + STEADY + WARMDOWN
# ============================================================================
def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step: int) -> float:
    """Muon momentum warms up from 0.85 → 0.95 over first 300 steps."""
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95

# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================
def find_latest_checkpoint():
    checkpoints = list(CKPT_DIR.glob('nstp_v3_step*.pt'))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]

def save_checkpoint(step, best_bpb, is_best=False):
    if step == 0:
        return
    try:
        ckpt = {
            'model': model.state_dict(),
            'muon_opt': muon_opt.state_dict(),
            'adamw_opt': adamw_opt.state_dict(),
            'step': step,
            'best_bpb': best_bpb,
            'config': config.__dict__,
            'history': history[-100:],
        }
        path = CKPT_DIR / f'nstp_v3_step{step:06d}.pt'
        torch.save(ckpt, path)
        if is_best:
            torch.save({'model': model.state_dict(), 'step': step, 'val_bpb': best_bpb},
                       SAVE_DIR / 'nstp_v3_best.pt')
        torch.save(ckpt, SAVE_DIR / 'nstp_v3_last.pt')
        # Keep only last 3 checkpoints
        old_ckpts = sorted(CKPT_DIR.glob('nstp_v3_step*.pt'), key=lambda p: p.stat().st_mtime)
        for old in old_ckpts[:-3]:
            old.unlink()
    except Exception as e:
        logger.error(f"Save FAILED at step {step}: {e}")

# ============================================================================
# EVALUATION
# ============================================================================
token_bytes_tensor = estimate_token_bytes(50257)

@torch.no_grad()
def evaluate():
    model.eval()
    model.reset_memory()
    val_bpb = compute_val_bpb(model, val_loader, DEVICE, vocab_size=50257,
                               token_bytes=token_bytes_tensor, max_eval_batches=40)
    val_ppl = compute_val_ppl(model, val_loader, DEVICE, vocab_size=50257, max_eval_batches=40)
    model.reset_memory()
    model.train()
    logger.info(f"  [eval] val_bpb: {val_bpb:.4f} | val_ppl: {val_ppl:.2f}")
    return val_bpb, val_ppl

# ============================================================================
# MAIN LOOP
# ============================================================================
logger.info("=" * 70)
logger.info("STARTING V3 TRAINING")
logger.info("=" * 70)

global_step = 0
best_bpb = float('inf')
history = []
tokens_seen = 0
start_time = time.time()

# Resume?
ckpt_path = find_latest_checkpoint()
if ckpt_path:
    logger.info(f"Resuming from {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        muon_opt.load_state_dict(ckpt['muon_opt'])
        adamw_opt.load_state_dict(ckpt['adamw_opt'])
        global_step = ckpt['step']
        best_bpb = ckpt.get('best_bpb', float('inf'))
        history = ckpt.get('history', [])
        logger.info(f"Resumed at step {global_step}, best_bpb={best_bpb:.4f}")
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        global_step = 0
        best_bpb = float('inf')
else:
    logger.info("Starting fresh")

train_iter = iter(train_loader)

try:
    while global_step < MAX_STEPS:
        # Determine LR
        progress = min(global_step / MAX_STEPS, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_mom = get_muon_momentum(global_step)

        # Apply to optimizers
        for group in muon_opt.param_groups:
            group['lr'] = LR_MUON * lrm
            group['momentum'] = muon_mom
        for group in adamw_opt.param_groups:
            group['lr'] = group['lr'] * lrm  # Will reset below

        model.reset_memory()

        # Accumulation loop
        accum_loss = 0.0
        for micro in range(ACCUM_STEPS):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x, return_drafts=False, softcap=SOFTCAP)
                logits = out['logits']
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            (loss / ACCUM_STEPS).backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()

        # Clip + step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        muon_opt.step()
        adamw_opt.step()

        # Zero grads (Muon doesn't have built-in zero_grad since it's custom)
        muon_opt.zero_grad()
        adamw_opt.zero_grad()

        global_step += 1

        # Logging
        if global_step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            tok_per_sec = tokens_seen / elapsed
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            cur_lr = LR_MUON * lrm
            logger.info(
                f"Step {global_step:5d}/{MAX_STEPS} | Loss: {accum_loss/ACCUM_STEPS:.4f} | "
                f"LR: {cur_lr:.2e} | Tok/s: {tok_per_sec:,.0f} | "
                f"VRAM: {vram_used:.1f}/{vram_total:.1f}GB | Time: {elapsed/60:.1f}m"
            )

        # Eval
        if global_step % EVAL_EVERY == 0:
            val_bpb, val_ppl = evaluate()
            elapsed = time.time() - start_time
            logger.info(f"  EVAL Step {global_step} | Time: {elapsed/60:.1f}m")
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                save_checkpoint(global_step, best_bpb, is_best=True)
                logger.info(f"    ✓ New best val_bpb: {best_bpb:.4f}")
            history.append({'step': global_step, 'val_bpb': val_bpb, 'val_ppl': val_ppl})

        # Save
        if global_step % SAVE_EVERY == 0:
            save_checkpoint(global_step, best_bpb)

        # GC management
        if global_step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif global_step % 5000 == 0:
            gc.collect()

        # Fast fail — use AVERAGED loss (accum_loss is sum over ACCUM_STEPS micro-steps)
        avg_loss = accum_loss / ACCUM_STEPS
        if math.isnan(avg_loss) or avg_loss > 100:
            logger.error(f"FAIL at step {global_step}: avg_loss={avg_loss:.4f} (accum={accum_loss:.2f})")
            exit(1)

except KeyboardInterrupt:
    logger.info("Interrupted — saving checkpoint")
    save_checkpoint(global_step, best_bpb)
except Exception as e:
    logger.error(f"Training error: {e}", exc_info=True)
    save_checkpoint(global_step, best_bpb)
    raise

logger.info("=" * 70)
logger.info("V3 TRAINING COMPLETE")
logger.info("=" * 70)
logger.info(f"val_bpb:          {best_bpb:.4f}")
logger.info(f"total_seconds:    {(time.time()-start_time):.1f}")
logger.info(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
logger.info(f"num_params_M:     {total_params / 1e6:.1f}")
logger.info(f"num_steps:        {global_step}")
logger.info(f"Best val_bpb:     {best_bpb:.4f}")
logger.info(f"Total time:       {(time.time()-start_time)/60:.1f}m")

save_checkpoint(global_step, best_bpb)
with open(SAVE_DIR / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
logger.info("✅ DONE")
