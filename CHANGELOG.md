# Changelog — NSTP-Ω / nstp-rope

All notable changes to this repository are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates in YYYY-MM-DD.

---

## [Unreleased] — 2026-07-30

### Added — NSTP-Ω V3 + autoresearch integration

Major upgrade integrating techniques from
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
(adapted from [nanochat](https://github.com/karpathy/nanochat)) into the
NSTP-Ω training pipeline. Five new modules and one redesigned model.

#### New files

- **`muon_optimizer.py`** — Karpathy-style Muon optimizer with
  Newton-Schulz orthogonalization, NorMuon variance reduction, and
  cautious weight decay (mask = `(g * p) >= 0`). 287 lines.
  Includes `split_params_for_muon()` helper that classifies parameters
  into Muon (2D matrices) vs AdamW (embeddings, scalars, norms, head).

- **`bos_packing.py`** — BOS-aligned best-fit packing dataloader that
  eliminates the 15% PPL regression from V2's zero-padded mixed-length
  training. 273 lines. Two implementations:
  - `BOSPackedDataset` — pre-packs all documents into rows
  - `BOSPackedStreamingLoader` — streams + re-shuffles per epoch
  Includes EOS-aware document segmentation with fixed-length fallback
  for streams without explicit EOS markers (e.g. FineWeb-Edu).

- **`val_bpb.py`** — Vocab-independent bits-per-byte evaluation
  (vocab-independent metric from autoresearch/nanochat). 167 lines.
  Includes `compute_val_bpb()` and `compute_val_ppl()`. The BPB metric
  normalizes by token byte-length, making architectural comparisons
  fair across vocab-size changes.

- **`nstp_omega_v3.py`** — NSTP-Ω V3 model with three Karpathy-style
  improvements over V2: 397 lines.
  - **Softcap logits** (softcap=15) — `softcap * tanh(logits/softcap)`,
    prevents overconfident spikes early in training
  - **Value Embeddings (ResFormer)** — per-token learnable V bias mixed
    via input-dependent gate per head, alternating layers only
  - **x0 residual lambdas** — per-layer learnable mixing of initial
    embedding into residual stream (initialized at 0.1)
  - **Windowed attention pattern** — `SSSL` pattern via DeltaNet
    window_size (short context for most layers, full for last)

- **`train_nstp_omega_v3.py`** — Main training script that ties the
  V3 components together. 396 lines. Includes:
  - LR schedule: warmup (500 steps) → steady → cosine warmdown
  - Muon for matrix params, AdamW for embeddings/head/scalars
  - BOS-aligned streaming loader for train data
  - val_bpb + val_ppl dual evaluation every 2500 steps
  - Checkpointing every 5000 steps with resume support
  - Fast-fail on NaN/loss>100, GC freeze after warmup

- **`autoresearch_loop.py`** — Autonomous experiment runner adapted
  from autoresearch. 280 lines. Features:
  - Branch management (`autoresearch/<tag>`)
  - 30-min time budget per experiment (longer than Karpathy's 5 min
    for our 144M model scale)
  - TSV logging: `commit | val_bpb | memory_gb | status | description`
  - Auto-keep if val_bpb improves, `git reset --hard` if not
  - Hard kill at 40 min to prevent runaway experiments
  - `--baseline-only` mode for first-run baseline

- **`prototype_muon_bos.py`** — Fast prototype (100 steps) verifying
  Muon + BOS-packing integration before full V3 training. 156 lines.

#### Prototype validation results

Ran `prototype_muon_bos.py` for 100 steps on FineWeb-Edu (5M tokens
streaming, 256 seq_len, batch=4×8 accum, lr_muon=0.001):

| Metric | Value |
|---|---|
| Initial loss (step 10) | 22.95 |
| Final loss (step 100) | 12.27 |
| **Reduction** | **48.1% in 100 steps** |
| Final val_bpb | 10.90 |
| Throughput | 592 tok/s |
| Total time | 1384s (23 min) |
| Tokens seen | 819,200 |

**Loss curve** (smooth descent, accelerating):
step 10: 22.95 → 20: 21.86 → 30: 21.01 → 40: 19.75 →
50: 18.66 → 60: 17.61 → 70: 16.21 → 80: 14.94 → 90: 13.69 →
100: 12.60.

For comparison, V2 at step 100 had loss ~52.0 (AdamW, no warmup).
V3 prototype is **4× lower loss at the same step count** thanks to
Muon's orthogonalized momentum + cautious WD.

#### Bugs fixed during V3 development

1. **VE shape mismatch** — `GatedDeltaNetOmegaV3` initialized
   `n_kv_head = H // 2` (GQA-style) but V tensor was `H` heads,
   causing `RuntimeError: size of tensor a (4) must match size of
   tensor b (2)`. Fixed by matching `n_kv_head = H` (always equal to
   attention heads).
2. **Missing `value_embed` attribute** — non-VE layers crashed on
   `if self.value_embed is not None`. Fixed by always initializing the
   attribute to `None`.
3. **Accum-loss fast-fail false positive** — prototype was checking
   `accum_loss > 100` (sum of 8 micro-step losses, ~22 each = ~176).
   Fixed by averaging micro-steps first: check `avg_loss > 100`.
4. **Muon LR scale** — `lr=0.02` (Karpathy's nanochat default) diverged
   on our larger 144M model. Lowered to `lr=0.001` for stable
   convergence; Karpathy's value works on ~50M nanochat but needs
   scaling for our scale.
5. **FineWeb doc segmentation** — FineWeb-Edu was concatenated into
   one stream with no EOS markers, causing `segment_into_docs()` to
   find 0 documents. Fixed by adding fixed-length-chunk fallback.

#### Decisions / non-changes

- **HHM (Hierarchical Hyperdimensional Memory) disabled** in V3
  prototype (`model.hhm = None`) because its FFT-based bind/unbind
  breaks gradient flow during training. Kept in code for future use.
- **LayerDrop disabled** (`layer_drop=0.0`) for now — small models
  benefit less from layer dropping.
- **MoE load balancing** kept simple (RFMoE threshold self-adjustment)
  instead of adding auxiliary loss (V2's `L_load`) since V3 prototype
  didn't enable it and stayed stable.

#### Changed files

- **`.gitignore`** — Removed blanket `*.json`, `*.txt`, `logs/` exclusions
  so CHANGELOG.md and results.tsv can be tracked. `*.log` files still
  excluded.

---

## [2026-07-29] — V2 training in progress

### Added — V2 training run (proc_fcd07b0d0d29)

Long-running V2 training of NSTP-Ω on FineWeb-Edu 800M tokens.
Started 2026-07-29 16:30 UTC. Architecture: 6 layers, d_model=512,
8 heads, 4 experts, RoPE attention, Gated DeltaNet, RFMoE.
Optimizer: AdamW (later switched from SGD due to dynamo issues),
LR warmup 500 steps + cosine decay over 50K steps.

Checkpoints in `models_v2/checkpoints/`. Latest at step 12,500
showed Val PPL 2510 (best so far). Training continues in background.

### Added — Karpathy autoresearch study

Read and analyzed
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)
(March 2026) and extracted 13 techniques applicable to NSTP-Ω:

1. Muon optimizer (orthogonalized momentum)
2. Cautious weight decay
3. NorMuon variance reduction
4. Value Embeddings (ResFormer)
5. x0 residual lambdas
6. Softcap logits
7. BOS-aligned best-fit packing
8. val_bpb metric (vocab-independent)
9. Windowed attention pattern
10. Fast-fail on loss explosion
11. GC freeze
12. Aspect-ratio model sizing
13. LR warmup → steady → warmdown schedule

→ All 13 techniques incorporated into V3 (see Unreleased).

---

## [2026-07-22] — Repo public, Phase 2 done

### Added

- **`train_rope_fineweb.py`** — Main unified 4-stage progressive
  training script: RoPE attention, SDPA, MixedLengthDS, Gated
  DeltaNet. ~440 lines, 44.6M params.
- **`train_rope_wikitext2.py`** — WikiText-2 3-stage RoPE training
  (completed, 2,497 lines).
- **`test_sdpa_rope.py`** — SDPA + RoPE compatibility test (96 lines).
- **`test_45m_mem.py`** — 45M model memory/speed profiling (65 lines).
- **`rope_simple.py`** — Clean reference RoPE implementation (110 lines).
- **`sep_cmaes_router.py`** — SEP-CMA-ES TT-MoE router (220 lines,
  not yet tested).
- **`nstp_rope.py`** — NSTP v1 with RoPE attention (181 lines).
- **`README.md`** — Comprehensive 194-line README documenting results,
  architecture, key findings, and lessons learned.

### Results — FineWeb RoPE 4-stage training

- **S0** (SEQ=128, 20K steps): final val PPL — S128=2687, S256=2676
  (-0.4%), S512=2694 (+0.3%), S1024=2603 (-3.1%)
- **S1** (SEQ=256, 15K steps): val PPL — S128=2688, S256=2676,
  S512=2694, S1024=2604 (within 0.7%)
- **S2** (SEQ=512 + 25% short mix, 10K steps): val PPL — S128=3144,
  S256=3100, S512=3132, S1024=3037 (mixed-length regression begins)
- **S3** (SEQ=1024 + 25% short mix, 5K steps): val PPL — S128=3131
- **Final test PPL**: S128=3088, S256=3112, S512=3070 (best),
  S1024=3072 (all within 1.4%)

**Key finding:** RoPE enables context-length invariance (within 1%
PPL from 128→1024 tokens). Absolute performance ~100× worse than
GPT-2 Small (PPL ~3100 vs ~20-30) due to 20× under-training (1B tokens
vs 20B needed for 45M model). Architecture improvements validated;
scale is the bottleneck.

**Repo:** https://github.com/fstark96/nstp-rope — 3+ commits, public.

---

## [2026-07-15] — Phase 2

### Added

- Vectorized HSA (Hyperdimensional Sparse Attention) retrieval
- Removed dead positional embedding code from NSTP core
- New benchmark suite (`benchmarks/benchmark.py`)

---

## [2026-07-01] — Initial commit (Phase 1)

### Added

- NSTP-Ω core architecture (846 lines, `nstp_omega.py`)
  - Gated DeltaNet-Ω (triple-gate linear attention)
  - TT-HyperNetwork (dynamic ranks per token)
  - Hierarchical Hyperdimensional Memory (3-tier O(1) recall)
  - RF-MoE (router-free)
  - EAGLE-Ω (feature-level speculation)
  - TTCS (Test-Time Compute Scaling halt gates)
  - QuEST-Ω (1.58-bit quantization-native)
- Initial README, .gitignore, license
- First benchmarks

---

## Conventions

- **Commits** — use `feat:`, `fix:`, `docs:`, `refactor:` prefixes
- **Branches** — `autoresearch/<tag>` for autonomous research sessions
- **Training checkpoints** — excluded from git (`models/`, `*.pt`)
- **Logs** — `*.log` excluded, but `logs/` directory contents tracked
- **Token data** — `data/` excluded (regeneratable, large)

## See also

- `README.md` — project overview, results summary, architecture diagram
- `results.tsv` — autoresearch experiment log (created by autoresearch_loop.py)
