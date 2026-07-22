"""
NSTP Core Package
Neuro-Symbolic Tensor Processor - Phase 1 Digital Implementation
"""

from .hsa import (
    HSAEncoder,
    HSAContextAccumulator,
    HSADenoiser,
    HyperdimensionalAttention,
    hyperdimensional_attention,
    cyclic_shift,
    bind,
    unbind,
    superposition,
    cosine_similarity_binary,
)

from .tt import (
    TTLinear,
    TTEmbedding,
    tt_decompose,
    tt_orthogonal_loss,
    reconstruct_tt,
)

from .moe import (
    TTCERRouter,
    TTCERExpert,
    TTCERMoE,
    top_k_routing,
    load_balancing_loss,
)

from .model import (
    NSTPConfig,
    NSTPBlock,
    NSTPModel,
)

from .losses import (
    NSTPLoss,
    hsa_denoising_loss,
    tt_orthogonal_loss,
    expert_balance_loss,
    specialization_loss,
)

__version__ = "1.0.0"
__all__ = [
    # HSA
    "HSAEncoder",
    "HSAContextAccumulator", 
    "HSADenoiser",
    "HyperdimensionalAttention",
    "hyperdimensional_attention",
    "cyclic_shift",
    "bind",
    "unbind",
    "superposition",
    "cosine_similarity_binary",
    # TT
    "TTLinear",
    "TTEmbedding",
    "tt_decompose",
    "tt_orthogonal_loss",
    "reconstruct_tt",
    "TTMatrix",
    # MoE
    "TTCERRouter",
    "TTCERExpert",
    "TTCERMoE",
    "top_k_routing",
    "load_balancing_loss",
    # Model
    "NSTPConfig",
    "NSTPBlock",
    "NSTPModel",
    # Losses
    "NSTPLoss",
    "hsa_denoising_loss",
    "expert_balance_loss",
    "specialization_loss",
]