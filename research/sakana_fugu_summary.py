"""
Sakana Fugu Architecture Summary — Extracted from papers
Key insights for NSTP improvement

## TRINITY (ICLR 2026)
- 0.6B SLM coordinator + ~20K learnable routing parameters
- sep-CMA-ES (derivative-free evolutionary optimization)
- Three roles: Thinker (strategy), Worker (execution), Verifier (check)
- Multi-turn: coordinator assigns different models at each turn
- Trained via evolutionary search, NOT gradient descent
- Result: 86.2% on LiveCodeBench (new SOTA at time of publication)
- Zero-shot generalization to unseen tasks (AIME, BigCodeBench, MT-Bench, GPQA)
- Key insight: "Evolution is uniquely suited to optimize tight, high-dimensional coordination 
  where traditional gradient-based methods fail"

## Conductor (ICLR 2026)  
- 7B Conductor model trained with RL (not supervised)
- Orchestrates a pool of frontier models: GPT-5, Gemini, Claude, open-source
- Outputs natural language: which agent, what subtask, what context to share
- Adaptive difficulty: simple questions → one model; complex coding → planner/coder/verifier pipeline
- "Recursive Test-Time Scaling": Conductor selects itself as worker, reads team output, 
  spins up corrective workflow on the fly
- Result: LiveCodeBench 83.9%, GPQA-Diamond 87.5%
- Outperforms multi-agent baselines at fraction of cost

## Fugu Product (June 2026)
- Three tiers: Fugu (balanced), Fugu Ultra (performance), Fugu Cyber (security)
- All via single OpenAI-compatible API
- Agent pool is swappable — can opt out specific providers
- Benchmark results (Fugu Ultra vs frontier):
  LiveCodeBench:     93.2 (vs GPT-5.5: 85.3, Opus 4.8: 87.8)
  GPQA-Diamond:      95.5 (vs GPT-5.5: 93.6, Opus 4.8: 92.0)
  SWE-Bench Pro:     73.7 (vs GPT-5.5: 58.6, Opus 4.8: 69.2)
  Humanity's Last:   50.0 (vs GPT-5.5: 41.4, Opus 4.8: 49.8)
  TerminalBench:     82.1 (vs GPT-5.5: 78.2, Opus 4.8: 74.6)

## Key Takeaways for NSTP
1. TRINITY + Conductor show that COORDINATION > MONOLITHIC SCALING
   - Small coordinator routing diverse experts beats single large model
   - NSTP's TT-MoE with learned routing is conceptually similar!
   
2. Test-time compute scaling (recursive self-orchestration) is a new scaling axis
   - Not just training data — runtime agent composition matters

3. sep-CMA-ES is highly effective for coordination problems
   - Evolutionary optimization can find routing strategies that gradient descent misses
   - Could replace gradient-based training for NSTP's TT-MoE router

4. Multi-turn role assignment (Thinker/Worker/Verifier)
   - NSTP could benefit from role-aware attention heads
   - Each head specializing in a different "role" in the computation
"""
print(__doc__[:2000])
