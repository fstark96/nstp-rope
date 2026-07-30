# NSTP-Ω Research Notes (Jul 2026) — Background Research

Quick reference for the latest ML techniques relevant to NSTP-Ω.
Compiled while V2 training runs (~20h to completion).

## A. Self-Healing / Adaptive Training

### A1. AdaMuon (arXiv 2507.11005, July 2025)
- Combines Muon orthogonal updates with Adam-like per-parameter adaptivity
- Adds (1) element-wise second momentum estimator on orthogonalized updates
- Adds (2) sign-stabilized orthogonal updates (momentum sign-transformed first)
- **Direct drop-in upgrade for our `muon_optimizer.py`**
- Link: https://arxiv.org/abs/2507.11005
- Code: (paper has reference impl)

### A2. Schedule-Free Muon (github.com/khurramkhalil/schedule-free-muon)
- PhD research: geometrically-aware optimizer combining Schedule-Free with Muon
- **Eliminates LR schedule hyperparameters entirely** (no warmup/cosine needed)
- Could replace our WARMUP_RATIO + WARMDOWN_RATIO logic
- Link: https://github.com/khurramkhalil/schedule-free-muon

### A3. DeadNeurons (github.com/Rekhii/DeadNeurons)
- Self-healing neural decoder — monitors hidden neurons, detects when they die, reinitializes
- **Direct fit for "self-healing" goal** — could be a small wrapper around our model
- Built in NumPy; we'd port to PyTorch for our architecture
- Link: https://github.com/Rekhii/DeadNeurons

## B. Self-Evolving / Dynamic Architectures

### B1. Mamba-3 (arXiv 2603.15569, March 2026)
- Successor to our Gated DeltaNet-Ω
- Three innovations: exponential-trapezoidal discretization, complex-valued state spaces, MIMO
- 1.5B scale: +0.6pp over Gated DeltaNet, +1.8pp with MIMO variant
- **Reference for V4 architecture upgrade**
- Link: https://arxiv.org/abs/2603.15569

### B2. "Don't Drop Dropout" (ICML 2026)
- Layer dropout / stochastic depth making comeback for LLM pretraining
- Higher accuracy + robustness to zero-shot layer pruning
- **Our `layer_drop` is currently disabled — should re-enable with proper schedule**
- Link: https://icml.cc/virtual/2026/poster/65775

### B3. ADEPT — Adaptive Dynamic Early-Exit (arXiv 2601.03700, Jan 2026)
- Early-exit transformers with dynamic exit heads
- Test-time compute scaling via repeated sampling/search
- **Direct match for our TTCS halt gates — currently we use them but don't fully exploit**
- Link: https://arxiv.org/abs/2601.03700

### B4. Awesome-Self-Evolving-Agents (github.com/XMUDeepLIT)
- Curated list of self-evolving agent papers/code
- Three dimensions: Model-Centric, Agentic, Environment
- Link: https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents

## C. Memory / Efficiency

### C1. FlashAttention-3 (Hopper-only)
- 1.6-1.8× speedup over FA2, FP8 support
- **SM90-only** — our RTX 4070 Ti SUPER is Ada (SM89), so we MUST use FA2
- Don't waste time trying to install FA3; stick with PyTorch SDPA EFFICIENT_ATTENTION

### C2. Sparse Distributed Memory / Beyond LLMs (arXiv 2604.11665, May 2026)
- Validates HDC/SDM as right direction for our HHM (Hierarchical Hyperdimensional Memory)
- "32GB as text file → HDC operations in million-token context"
- **Our HHM is on the right track — need to re-enable it (was disabled due to memory issues)**

## D. Architectural

### D1. Gated DeltaNet (ICLR 2025)
- The foundation our NSTP-Ω is built on
- Outperforms Mamba2 and DeltaNet across language modeling, retrieval, length extrapolation
- Hybrid: Gated DeltaNet + sliding window attention or Mamba2 layers
- **Confirms our architectural direction**
- Link: https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf

### D2. RWKV-7
- Linear attention with value residual + delta-rule update
- Part of the broader trend away from softmax attention
- Worth studying for future versions

## D. Implementation Suggestions for train_nstp_omega_v3.py

1. **Switch to AdaMuon** — replace `Muon` in `muon_optimizer.py` with AdaMuon variant (one paper)
2. **Add schedule-free option** — env var SCHEDULE_FREE=1 skips warmup/cosine entirely
3. **Re-enable layer dropout** — `layer_drop=0.05` with cosine schedule (matches "Don't Drop Dropout")
4. **Self-healing wrapper** — detect dead ReLU/GeLU units (output always 0), reinit weights
5. **Better eval early-exit** — enable TTCS halt gates during eval for inference speed
6. **HHM v2** — fix memory issue blocking HHM, re-enable hierarchical hyperdimensional memory
7. **Mamba-3 inspired** — replace `erase_gate = sigmoid(...)` with exp-trapezoidal update rule
8. **Self-evolving layer growth** — every N steps, try adding 1 layer; keep if val improves
9. **ReLoRA for FFN** — replace dense FFN with low-rank updates (B@A), 95% memory savings

## F. Next Experiment Ideas

Based on queue results (best: lr_muon=0.005), try:
- V4: lr_muon=0.005 + AdaMuon improvements
- V4: schedule-free Muon (no warmup/cosine)
- V4: re-enabled layer dropout with cosine schedule
- V4: self-healing dead-neuron detection + reinit
- V4: HHM v2 (fixed memory)
- V4: ReLoRA-style FFN — massive VRAM savings
- V4: try growing network (start small, add layers if val improves)

## G. Pitfalls / Notes

- FA3 won't work on RTX 4070 Ti SUPER (SM89 Ada, not SM90 Hopper)
- V2 already crashed once from concurrent GPU access — never run V3 + V2 simultaneously
- Muon LR sweet spot is ~0.005 for our 144M model (not 0.001, not 0.02)
- Mixed-length zero-padding caused V2 regression → use BOS-packing (already in V3)

## H. Top GitHub Repos (for cloning/study)

1. **flash-linear-attention** (github.com/fla-org/flash-linear-attention)
   - Official hardware-efficient GatedDeltaNet, RWKV, Mamba kernels
   - Platform-agnostic (NVIDIA/AMD/Intel)
   - PyPI package: `pip install flash-linear-attention`
   - **Use their GatedDeltaNet layer as reference for V4**

2. **Muon** (github.com/KellerJordan/Muon)
   - Original Muon optimizer implementation by Keller Jordan
   - Polar-express coefficients, Newton-Schulz iteration
   - **Mirror our muon_optimizer.py against this for validation**

3. **nanochat** (github.com/karpathy/nanochat)
   - Already studied; full reference impl we used for V3 base
   - Has Muon, value embeddings, val_bpb

4. **autoresearch** (github.com/karpathy/autoresearch)
   - Already studied; our autoresearch_loop.py mirrors this design

5. **AutoResearchClaw** (github.com/aiming-lab/AutoResearchClaw)
   - "Fully autonomous & self-evolving research from idea to paper"
   - NaN/Inf fast-fail, self-healing repair, iterative refinement (up to 10 rounds)
   - **Reference for self-healing autoresearch loop**

6. **Awesome-Self-Evolving-Agents** (github.com/XMUDeepLIT)
   - Curated list of self-evolving agent papers/code

## I. Self-Evolving Specific (Network Growth)

From "Growing Neural Networks" (arXiv 2501.18012, Jan 2025):
- Method 1: Auxiliary weight that directly controls network size
- Method 2: Controller-generated mask to modulate neuron participation
- Both optimize size via gradient descent
- **For NSTP-Ω: every K steps, spawn a candidate extra layer; keep it for K more steps if val improves**

## J. Self-Healing Specific (Dead Neurons)

From "DeadNeurons" (github.com/Rekhii) and standard practice:
- Monitor: track fraction of neurons outputting exactly 0 across all batches
- If dead_fraction > 0.5: reinitialize that neuron's weights (Kaiming init)
- Cheap to compute (just one forward pass per K training steps)
- **Could add to our `train_nstp_omega_v3.py` as `self_healing.py` helper**
