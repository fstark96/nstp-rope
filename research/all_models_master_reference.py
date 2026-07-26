"""
NSTP MASTER REFERENCE — All 10 Frontier Models Deep Dive
Compiled July 25, 2026
What each model does + what NSTP can learn from it
"""

FRONTIER_MODELS = {
    "Claude Fable 5": {
        "provider": "Anthropic",
        "released": "June 9, 2026",
        "class": "Mythos (highest capability tier)",
        "params": "NOT disclosed (estimated >1T)",
        "context": "1M tokens",
        "pricing": "$10/$50 per MTok",
        "open_weight": False,
        "benchmarks": {
            "SWE-bench Verified": "95.0%",
            "SWE-bench Pro": "80.3%",
            "CursorBench": "#1",
            "Hebbia Finance": "#1",
            "BenchAlign": 82.76,
        },
        "key_innovations": [
            "Classifier-based dual release (same model, different classifiers)",
            "Long-horizon agentic workflows (multi-hour without degradation)",
            "Persistent memory: 3× more improvement than Opus 4.8",
            "Vision SOTA: rebuilds source code from screenshots",
            "Scientific reasoning: novel hypothesis generation",
            "Constitutional AI + RLHF alignment",
        ],
        "nstp_lessons": [
            "Add persistent episodic memory to HDC bind/unbind",
            "Train on multi-step task completion (not just next-token)",
            "Classifier-gated routing: small classifier decides expert per input",
            "Self-correction loops: model evaluates own output and re-routes",
        ],
    },

    "Claude Mythos 5": {
        "provider": "Anthropic",
        "released": "June 9, 2026 (restricted to approved partners)",
        "class": "Mythos (same as Fable 5, no safety classifiers)",
        "params": "NOT disclosed (same as Fable 5)",
        "context": "1M tokens",
        "pricing": "$10/$50 per MTok",
        "open_weight": False,
        "benchmarks": {
            "SWE-bench Verified": "93.9%",
            "SWE-bench Pro": "77.8%",
            "BenchAlign": 83.01,
        },
        "key_innovations": [
            "Same architecture as Fable 5, no safety restrictions",
            "Strongest cybersecurity capabilities of any model",
            "Novel scientific hypotheses: 80% preference over Opus",
            "Drug design: 10× acceleration, matched human operators",
            "Genomics research: outperformed Science paper model (100× smaller)",
        ],
        "nstp_lessons": [
            "Safety classifiers can be added post-training (not baked into architecture)",
            "Same architecture, different deployment = massive flexibility",
            "NSTP could add safety routing as a separate module",
        ],
    },

    "GPT-5.6": {
        "provider": "OpenAI",
        "released": "July 9, 2026",
        "class": "Three tiers: Sol (frontier), Terra (balanced), Luna (fast/cheap)",
        "params": "NOT disclosed",
        "context": "1M tokens (1.05M for Terra)",
        "pricing": "Sol: $5/$30, Terra: $2.50/$15, Luna: $1/$3",
        "open_weight": False,
        "benchmarks": {
            "BenchAlign (Sol)": 81.46,
            "BenchAlign (Terra)": 71.99,
            "LiveCodeBench": "~85%",
            "SWE-bench Pro": "64.6% (Sol)",
        },
        "key_innovations": [
            "Tri-model architecture: Sol/Terra/Luna for different use cases",
            "Programmatic Tool Calling: writes and runs programs in-memory",
            "Zero Data Retention (ZDR) compatible",
            "Long-horizon reasoning: sustained multi-hour tasks",
            "Warmer, more natural tone (vs older robotic GPT-5)",
        ],
        "nstp_lessons": [
            "Tri-model approach: could have NSTP-Small/Medium/Large for different tasks",
            "Programmatic tool calling: model writes code to solve subproblems",
            "Cost tiers matter: different models for different latency/cost budgets",
        ],
    },

    "Kimi K3": {
        "provider": "Moonshot AI (China)",
        "released": "July 16, 2026",
        "class": "Open-weight frontier (2.8T params)",
        "params": "2.8T total, MoE",
        "context": "1,048,576 tokens (1M)",
        "pricing": "$3/$15 per MTok",
        "open_weight": True,
        "benchmarks": {
            "BenchAlign": 79.98,
            "Frontend Code Arena": "#1 (beats Fable 5)",
            "Long-horizon knowledge work": "Elo 1547 (behind only Fable 5)",
        },
        "key_innovations": [
            "Kimi Delta Attention (KDA): linear attention, 6.3× faster decoding at 1M tokens",
            "Attention Residuals (AttnRes): smoother info flow in deep models",
            "Latent-space routing: MoE routing in latent space",
            "Quantile Balancing: load management across experts",
            "Soft dropping: overflow tokens handled gracefully",
            "2.5× scaling efficiency improvement over Kimi K2",
        ],
        "nstp_lessons": [
            "CRITICAL: KDA is linear attention — replaces softmax with efficient linear operation",
            "Attention Residuals: add residual connections between attention layers",
            "Latent-space routing: route in compressed space, not raw hidden states",
            "Soft dropping: handle MoE overflow gracefully instead of hard dropping",
            "2.5× efficiency gain from architectural changes alone",
        ],
    },

    "Claude Opus 4.8": {
        "provider": "Anthropic",
        "released": "May 28, 2026",
        "class": "Opus (high-end, below Mythos)",
        "params": "NOT disclosed",
        "context": "1M tokens",
        "pricing": "$5/$25 per MTok",
        "open_weight": False,
        "benchmarks": {
            "SWE-bench Verified": "88.6%",
            "SWE-bench Pro": "69.2%",
            "GPQA Diamond": "93.6%",
            "BenchAlign": 77.44,
        },
        "key_innovations": [
            "4× less likely to let code flaws pass unremarked vs predecessor",
            "Strong coding agent performance",
            "Consistent long-running work handling",
            "Improved judgment quality for enterprise workflows",
        ],
        "nstp_lessons": [
            "Self-verification: model checks its own code for flaws",
            "Consistency in long-running tasks is a key differentiator",
            "Judgment quality improves with scale (not just raw capability)",
        ],
    },

    "Muse Spark 1.1": {
        "provider": "Meta (Superintelligence Labs)",
        "released": "July 2026",
        "class": "Multimodal reasoning model",
        "params": "NOT disclosed",
        "context": "1M tokens",
        "pricing": "Not listed (Meta API)",
        "open_weight": False,
        "benchmarks": {
            "BenchAlign": 76.60,
            "Coding": "Strong (exact numbers TBD)",
            "Agentic tasks": "Major gains in tool/computer use",
        },
        "key_innovations": [
            "Multimodal reasoning: text + image + video input",
            "Agentic task completion with tool use",
            "Computer use: cursor control, file system interaction",
            "Multi-agent reasoning capabilities",
            "Token efficiency improvements",
        ],
        "nstp_lessons": [
            "Multimodal: extend HDC bind/unbind to visual embeddings",
            "Computer use: model interacts with desktop environment",
            "Token efficiency: do more with fewer tokens (important for small models)",
        ],
    },

    "Grok 4.5": {
        "provider": "xAI (Elon Musk)",
        "released": "July 8, 2026 (public), June 28 (private beta)",
        "class": "V9 foundation model (1.5T params MoE)",
        "params": "1.5T total (3× larger than V8-small's 500B)",
        "context": "500K tokens",
        "pricing": "$2/$6 per MTok",
        "open_weight": False,
        "benchmarks": {
            "BenchAlign": 75.55,
            "Speed": "233 tok/s (fastest measured)",
            "Context": "2M (estimated largest useful context)",
        },
        "key_innovations": [
            "V9 ground-up redesign (not incremental from V8)",
            "Trained with Cursor AI coding platform data",
            "Colossus supercomputer training",
            "Fastest inference: 233 tok/s",
            "RLHF + Grok Build improvements",
        ],
        "nstp_lessons": [
            "Ground-up redesign beats incremental improvements",
            "Training on real coding platform data (Cursor) = better code understanding",
            "Speed matters: 233 tok/s enables real-time agentic workflows",
        ],
    },

    "Gemini 3.6 Flash": {
        "provider": "Google",
        "released": "July 21, 2026",
        "class": "Flash (efficiency-focused)",
        "params": "NOT disclosed",
        "context": "1M tokens",
        "pricing": "$1.50/$7.50 per MTok",
        "open_weight": False,
        "benchmarks": {
            "BenchAlign": 75.54,
            "Efficiency": "Best quality-per-dollar in Flash tier",
        },
        "key_innovations": [
            "Efficiency-optimized: best quality at lowest cost",
            "1M context at Flash-tier pricing",
            "Multimodal: text + image + video",
            "Google's efficiency focus: reducing token cost",
        ],
        "nstp_lessons": [
            "Efficiency matters more than raw capability for many use cases",
            "Small models can compete if they're efficient enough",
            "Cost-per-quality is a key metric, not just raw quality",
        ],
    },

    "Qwen3.7 Max": {
        "provider": "Alibaba",
        "released": "May 20, 2026",
        "class": "Flagship reasoning model",
        "params": "~1.6T MoE",
        "context": "1M tokens (984K actual)",
        "pricing": "$2.50/$7.50 per MTok",
        "open_weight": False,
        "benchmarks": {
            "SWE-bench Pro": "80.4%",
            "Terminal-Bench": "#1 (beats Opus 4.6)",
            "MCP-Atlas": "#1",
            "BenchAlign": 71.91,
        },
        "key_innovations": [
            "Chain-of-thought reasoning architecture",
            "MCP tool orchestration (Model Context Protocol)",
            "Long-horizon agentic workloads: hours, not seconds",
            "Agent-first design: purpose-built for autonomous tasks",
        ],
        "nstp_lessons": [
            "MCP integration: standardized tool calling protocol",
            "Chain-of-thought: explicit reasoning steps before action",
            "Agent-first: design for autonomous multi-step workflows",
        ],
    },

    "MiniMax M3": {
        "provider": "MiniMax (Shanghai)",
        "released": "June 1, 2026",
        "class": "Open-weight frontier (428B total / 23B active)",
        "params": "428B total, ~23B active per token",
        "context": "1,048,576 tokens (1M)",
        "pricing": "$0.30/$1.20 per MTok",
        "open_weight": True,
        "benchmarks": {
            "SWE-bench Verified": "80.5%",
            "BenchAlign": 68.80,
            "Coding": "Frontier-level",
            "Agentic": "Frontier-level",
        },
        "key_innovations": [
            "MiniMax Sparse Attention (MSA): 28.4× less compute at 1M context",
            "14.2× prefill speedup, 7.6× decoding speedup on H800",
            "Native multimodal: text + image + video input",
            "First open-weight model to unite frontier coding + 1M context + multimodal",
            "Blockwise sparse attention mechanism",
        ],
        "nstp_lessons": [
            "CRITICAL: MSA is 28.4× more efficient than GQA at 1M context",
            "Blockwise sparse attention: only attend to relevant blocks",
            "Open-weight at $0.30/$1.20: proves efficiency enables accessibility",
            "NSTP could adopt sparse attention patterns from MSA",
        ],
    },

    "GLM-5.1": {
        "provider": "Z.AI (Zhipu AI)",
        "released": "June 2026",
        "class": "Open-weight agentic model (744B total / 40B active)",
        "params": "744B total, 256 routed + 1 shared experts, 8+1 active per token",
        "context": "203K tokens",
        "pricing": "$1.40/$4.40 per MTok",
        "open_weight": True,
        "benchmarks": {
            "SWE-bench Pro": "58.4%",
            "BenchAlign": 66.94,
            "Agentic": "8-hour continuous autonomous work",
        },
        "key_innovations": [
            "GlmMoeDSA architecture: Gated DeltaNet linear attention + standard attention + sparse MoE FFN",
            "Gated DeltaNet: linear attention variant with gating mechanism",
            "Hybrid attention: linear + standard in same model",
            "78 layers with 256 routed + 1 shared expert per MoE layer",
            "8-hour continuous autonomous task completion",
        ],
        "nstp_lessons": [
            "CRITICAL: Hybrid linear + standard attention in same model",
            "Gated DeltaNet: gated linear attention for efficiency",
            "Shared expert: 1 expert always active (provides baseline capability)",
            "NSTP could add a shared expert to MoE for baseline + specialized routing",
        ],
    },
}

# Summary of what NSTP can learn from ALL models
NSTP_IMPROVEMENTS = {
    "Architecture": [
        "KDA (Kimi K3): Linear attention replacing softmax — 6.3× faster at 1M tokens",
        "MSA (MiniMax M3): Blockwise sparse attention — 28.4× less compute at 1M context",
        "Gated DeltaNet (GLM-5.1): Hybrid linear + standard attention",
        "Attention Residuals (Kimi K3): Smoother info flow in deep models",
        "Shared expert (GLM-5.1): 1 expert always active for baseline capability",
        "Soft dropping (Kimi K3): Graceful overflow handling in MoE",
    ],
    "Training": [
        "Agentic training data (Fable 5): Multi-step task completion + error recovery",
        "Coding platform data (Grok 4.5): Train on real developer workflows",
        "Chain-of-thought (Qwen3.7): Explicit reasoning steps before action",
        "Chinchilla scaling: 20 tokens/param optimal (we have 2.5× sub-optimal)",
    ],
    "Routing": [
        "Latent-space routing (Kimi K3): Route in compressed space, not raw hidden states",
        "Quantile Balancing (Kimi K3): Better load management across experts",
        "Classifier-gated routing (Fable 5): Small classifier decides expert per input",
        "Shared expert (GLM-5.1): Always-active baseline + specialized routing",
    ],
    "Memory": [
        "Persistent episodic memory (Fable 5): 3× more improvement than Opus 4.8",
        "File-based memory (Fable 5): Model writes and reads its own notes",
        "Working memory: Maintain persistent hidden state across forward passes",
    ],
    "Efficiency": [
        "Cost tiers (GPT-5.6): Different models for different latency/cost budgets",
        "Token efficiency (Muse Spark 1.1): Do more with fewer tokens",
        "Speed (Grok 4.5): 233 tok/s enables real-time agentic workflows",
        "Open-weight efficiency (MiniMax M3): Frontier capability at $0.30/$1.20",
    ],
    "Safety": [
        "Classifier-based dual release (Fable 5): Same model, different classifiers",
        "Self-verification (Opus 4.8): Model checks its own output for flaws",
        "Constitutional AI (Fable 5): Predefined rules for ethical decision-making",
    ],
}

# Priority ranking for NSTP improvements
PRIORITY_IMPROVEMENTS = [
    {
        "rank": 1,
        "improvement": "Scale training data to 780M+ tokens",
        "source": "Chinchilla scaling laws",
        "impact": "HIGH — currently 7.8× sub-optimal",
        "difficulty": "MEDIUM — need more data + compute",
    },
    {
        "rank": 2,
        "improvement": "Add linear attention component (KDA-style)",
        "source": "Kimi K3 (6.3× faster), GLM-5.1 (Gated DeltaNet)",
        "impact": "HIGH — efficiency + longer context",
        "difficulty": "HIGH — architectural change",
    },
    {
        "rank": 3,
        "improvement": "Add shared expert to MoE",
        "source": "GLM-5.1 (256 routed + 1 shared)",
        "impact": "MEDIUM-HIGH — baseline capability + specialization",
        "difficulty": "LOW — add one always-active expert",
    },
    {
        "rank": 4,
        "improvement": "Latent-space routing for MoE",
        "source": "Kimi K3",
        "impact": "MEDIUM — better routing decisions",
        "difficulty": "MEDIUM — change routing space",
    },
    {
        "rank": 5,
        "improvement": "Soft dropping for MoE overflow",
        "source": "Kimi K3",
        "impact": "MEDIUM — graceful degradation",
        "difficulty": "LOW — modify overflow handling",
    },
    {
        "rank": 6,
        "improvement": "Attention Residuals between layers",
        "source": "Kimi K3",
        "impact": "MEDIUM — smoother info flow",
        "difficulty": "LOW — add residual connections",
    },
    {
        "rank": 7,
        "improvement": "Blockwise sparse attention (MSA-style)",
        "source": "MiniMax M3 (28.4× efficiency)",
        "impact": "HIGH — massive efficiency gain",
        "difficulty": "HIGH — new attention mechanism",
    },
    {
        "rank": 8,
        "improvement": "Persistent episodic memory",
        "source": "Claude Fable 5 (3× improvement)",
        "impact": "HIGH — long-horizon capability",
        "difficulty": "MEDIUM — extend HDC bind/unbind",
    },
    {
        "rank": 9,
        "improvement": "Chain-of-thought reasoning",
        "source": "Qwen3.7 Max",
        "impact": "MEDIUM — explicit reasoning steps",
        "difficulty": "LOW — training data change",
    },
    {
        "rank": 10,
        "improvement": "Classifier-gated routing",
        "source": "Claude Fable 5",
        "impact": "MEDIUM — task-specific expert selection",
        "difficulty": "LOW — add small classifier",
    },
]

if __name__ == "__main__":
    print("=" * 70)
    print("NSTP MASTER REFERENCE — 10 Frontier Models Deep Dive")
    print("=" * 70)
    
    for name, info in FRONTIER_MODELS.items():
        print(f"\n{'─' * 70}")
        print(f"📌 {name} ({info['provider']})")
        print(f"   Params: {info['params']} | Context: {info['context']} | Open: {info['open_weight']}")
        print(f"   Pricing: {info['pricing']}")
        print(f"   Key innovations:")
        for inn in info['key_innovations'][:3]:
            print(f"     • {inn}")
        print(f"   NSTP lessons:")
        for lesson in info['nstp_lessons'][:2]:
            print(f"     → {lesson}")
    
    print(f"\n{'=' * 70}")
    print("TOP 10 PRIORITY IMPROVEMENTS FOR NSTP")
    print(f"{'=' * 70}")
    for imp in PRIORITY_IMPROVEMENTS:
        print(f"\n#{imp['rank']}: {imp['improvement']}")
        print(f"   Source: {imp['source']}")
        print(f"   Impact: {imp['impact']}")
        print(f"   Difficulty: {imp['difficulty']}")
