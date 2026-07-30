# Autoresearch Experiments — NSTP-Ω V3

Each experiment runs `train_nstp_omega_v3.py` for a fixed time budget
and measures `val_bpb`. Results logged to `results.tsv`.

Baseline (already done in prototype, 100 steps):
- val_bpb: **10.90** (loss 12.60 at step 100, 592 tok/s)

## Experiment Queue (priority order)

### 1. BASELINE V3 — full proto conditions
- lr_muon=0.001, lr_adamw=0.001
- SEQ_LEN=256, batch=4×8 accum
- 5M tokens of FineWeb streaming
- 100 steps
- **Expected: val_bpb ~10.90** (confirms our prototype result)

### 2. INCREASE MUON LR
- Same as baseline but lr_muon=0.005 (5×)
- Hypothesis: Muon can handle higher LR with longer warmup
- **Expected: faster initial descent, maybe lower final val_bpb**

### 3. WARMUP EXTENSION
- Same as baseline but warmup_ratio=0.05 (5% = ~5 steps, gentle)
- Compare: does longer warmup stabilize early training?

### 4. LARGER BATCH
- SEQ_LEN=512, batch=2×16 accum (same effective tokens)
- Test: does longer context help Muon converge?

### 5. SOFTCAP TOGGLE
- Same as baseline but softcap=0 (disabled)
- Test: does softcap actually help or hurt?

### 6. WINDOW PATTERN VARIANT
- Change window pattern from SSSL to LLLL (full context always)
- Test: windowed attention vs full attention cost/benefit

### 7. VALUE EMBEDDING TOGGLE
- Disable Value Embeddings (set has_ve=False always)
- Test: do VEs actually help? Cheap ablation

### 8. ADAMW ONLY
- Replace Muon with AdamW (everything)
- **Expected: WORSE val_bpb** — confirms Muon is helping

## Running Order

Run experiments 1-7 in sequence. Each takes ~23 min.
Total: ~3 hours for 8 experiments.

## Results Format

After each experiment, results.tsv appends:
```
commit	val_bpb	memory_gb	status	description
```
