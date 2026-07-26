"""
NSTP Complete Research Notes — July 25, 2026
All findings from this session, saved for future reference.

## Session Summary
- Verified NSTP PPL=3.82 on WikiText-2 (SEQ=128) via 6 sanity checks
- ChatGPT confirmed: "No obvious sign perplexity is artificially low"
- Discovered NSTP can't learn from scratch on diverse text (FineWeb-Edu)
- Transfer learning WikiText-2 → FineWeb-Edu works (PPL 16K → 1,765)
- Researched Sakana Fugu (TRINITY + Conductor) architecture
- Deep-dived Claude Fable 5 (Mythos-class, 95% SWE-bench)
- Compiled frontier model landscape (July 2026)

## Key Findings

### 1. NSTP Architecture Assessment
- WORKS: PPL=3.82 on WikiText-2, 567× better than standard TF
- LIMITATION: Can't learn from scratch on diverse web text
- STRENGTH: HDC bind/unbind is genuinely novel (no comparable paper)
- GAP: 25,000× parameter gap vs Claude Fable 5

### 2. Sakana Fugu Insights
- TRINITY: 0.6B coordinator + <20K routing params, sep-CMA-ES
- Conductor: 7B RL orchestrator, recursive self-correction
- Key: Small coordinator routing diverse experts > single large model
- Application: NSTP's TT-MoE is conceptually aligned

### 3. Claude Fable 5 Insights
- Mythos-class, same model as Mythos 5 + safety classifiers
- 95% SWE-bench Verified, 1M context
- Agentic training: multi-step + error recovery + self-correction
- Persistent memory: 3× more improvement than Opus 4.8
- Application: Add episodic memory, agentic training data

### 4. Chinchilla Scaling
- 39M params → optimal 780M tokens
- We have 100M → 7.8× sub-optimal
- Need more data or smaller model

### 5. Frontier Model Landscape (July 2026)
- Claude dominates (Opus 5 #1, Mythos 5 #2, Fable 5 #3)
- Open-weight closing fast (Kimi K3 #5, MiniMax M3 #18)
- DeepSeek V4: 1.6T MoE, 80.6% SWE-bench, MIT license

## Files Created
- /tmp/nstp-v2/research/sakana_fugu_summary.py
- /tmp/nstp-v2/research/improvement_roadmap.py
- /tmp/nstp-v2/extract_fable5.py
- /tmp/nstp-v2/extract_benchmarks.py
- /tmp/nstp-v2/train_finetune.py (running, step 25K, PPL=1,765)
- /tmp/nstp-v2/data/fineweb_*_tokens.npy (100M tokens)

## Skills Created
- sakana-fugu-nstp-research (research category)
- nstp-frontier-models-research (research category)
"""
print(__doc__)
