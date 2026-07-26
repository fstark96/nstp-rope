"""
Research Summary: NSTP Improvement Roadmap — Key Findings

## 1. HDC + Transformer Research
- **HDT (Hyperdimensional Transformer)**: Replaces softmax attention with integer dot-products 
  in hyperdimensional space. Thresholded to form Boolean mask (no softmax needed). Uses 
  Hamming distance for similarity. Published at COLM 2024. Key insight: HDC attention is 
  compute-efficient and can handle long sequences without quadratic scaling.
- **NSTP's VH.bind/unbind is unique**: No existing paper does FFT-based HDC position binding 
  with continuous (not binary) representations. This is a genuine novel contribution.
- **ScalableHD (2025)**: HDC for scalable high-throughput computing. Shows HDC scales better 
  than traditional approaches for parallel operations.
- **Medium article on HDC attention**: Combines symbolic binding with neural attention — 
  validates our neuro-symbolic approach.

## 2. Chinchilla Scaling Laws (for 39M params)
- **Optimal: 15-25 tokens per parameter** (consensus: ~20 tokens/param)
- For NSTP 39M: optimal training = 780M tokens
- Our current 100M tokens = ~2.5 tokens/param → **4x sub-optimal**
- Revised Chinchilla (2024): accounts for inference cost — favors slightly smaller models 
  trained longer
- **Implication**: With 100M tokens, our 39M model is over-parameterized. Either:
  (a) Scale to 780M+ tokens for Chinchilla optimal, OR
  (b) Reduce model to ~4M params for current data, OR
  (c) Accept sub-optimal but see if HDC/MoE compensates

## 3. MoE Load Balancing Best Practices
- **Auxiliary loss weight**: 0.01 (standard)
- **Router z-loss**: 0.001 (stabilizes routing, prevents collapse)
- **Monitoring**: Track routing entropy, expert utilization, dropped tokens
- **Capacity factor**: 1.25 (standard)
- **Key finding**: Without load balancing, experts collapse — same token always routed to 
  same expert. This is already in our code (aux_loss_coef=0.01).
- **Expert utilization metric**: Should be uniform (1/num_experts per expert). If one expert 
  handles >50% of tokens, routing has collapsed.

## 4. RoPE / Positional Encoding
- **ALiBi**: Position bias added to attention scores. No learned parameters. Better 
  extrapolation than learned absolute.
- **YaRN**: Extends RoPE to longer contexts via interpolation.
- **For NSTP**: Our VH.bind/unbind is the positional encoding. The challenge is that 
  positions are bound at encoding time and can't easily be changed. 
  **Recommendation**: Keep VH.bind for now but experiment with learned positional 
  modulation (add a small learned vector to the position before binding).

## 5. Sakana Fugu — Key Architectural Insight
TRINITY + Conductor show that **small coordinator routing diverse experts** beats single 
large model. NSTP's TT-MoE is conceptually similar — could adopt:
1. **sep-CMA-ES for router**: Evolutionary optimization instead of gradient descent 
   for routing decisions
2. **Role-based specialization**: Different expert heads for different "roles" (like 
   Thinker/Worker/Verifier)
3. **Recursive self-correction**: Model evaluates its own output and re-routes

## Priority Recommendations
1. **Scale data to 780M+ tokens** (Chinchilla optimal for 39M params)
2. **Add router z-loss** (0.001) to prevent expert collapse
3. **Monitor expert utilization** — add logging during training
4. **Test ALiBi-style positional bias** — small addition to VH.bind
5. **Experiment with learned routing weights** — evolutionary vs gradient
6. **Benchmark on LM Evaluation Harness** — HellaSwag, MMLU, PIQA, GSM8K

## Key Files
- Sakana Fugu summary: C:/Users/user/AppData/Local/Temp/nstp-v2/research/sakana_fugu_summary.py
- HDT paper: https://www.emergentmind.com/topics/hyperdimensional-transformer-hdt
- Conductor paper: https://arxiv.org/abs/2512.04388
- TRINITY paper: https://arxiv.org/abs/2512.04695
- Chinchilla revised: https://www.educatingsilicon.com/2024/04/29/revised-chinchilla-scaling-laws/
"""
print(__doc__[:3000])
