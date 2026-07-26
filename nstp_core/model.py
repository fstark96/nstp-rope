"""
NSTP Model Architecture
Combines HSA (Hyperdimensional Symbolic Attention) + TT-CER MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

from .hsa import HyperdimensionalAttention, HSAEncoder, HSAContextAccumulator, HSADenoiser
from .moe import TTCERMoE, TTCERExpert
from .tt import TTLinear, TTEmbedding, tt_orthogonal_loss


@dataclass
class NSTPConfig:
    """Configuration for NSTP Model."""

    # Model dimensions
    vocab_size: int = 128256
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8

    # HSA Configuration
    hsa_dim: int = 16384
    hsa_bind_mode: str = "xor"
    hsa_binary: bool = True
    hsa_denoise_iterations: int = 3
    hsa_trainable_encoder: bool = True

    # TT-CER MoE Configuration
    num_experts: int = 8
    top_k: int = 2
    d_ff: int = 2048
    router_tt_ranks: List[int] = field(default_factory=lambda: [1, 16, 16, 1])
    expert_tt_ranks: List[int] = field(default_factory=lambda: [1, 16, 16, 16, 1])
    capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01

    # TT Embedding
    use_tt_embedding: bool = True
    embedding_tt_ranks: List[int] = field(default_factory=lambda: [1, 16, 16, 1])

    # Training
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1

    # Loss coefficients
    hsa_denoise_loss_coef: float = 0.1
    tt_ortho_loss_coef: float = 0.01
    expert_balance_loss_coef: float = 0.01
    specialization_loss_coef: float = 0.0

    # Optimization
    max_seq_len: int = 1_000_000

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        assert self.hsa_dim % self.num_heads == 0, "hsa_dim must be divisible by num_heads"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"


class NSTPBlock(nn.Module):
    """
    Single NSTP Transformer Block.
    Structure:
    1. HSA Attention (replaces standard MHA)
    2. TT-CER MoE (replaces standard FFN)
    3. Residual connections + LayerNorm
    """

    def __init__(self, config: NSTPConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # HSA Attention
        self.attention = HyperdimensionalAttention(
            d_model=config.d_model,
            hsa_dim=config.hsa_dim,
            num_heads=config.num_heads,
            binary=config.hsa_binary,
            bind_mode=config.hsa_bind_mode,
            denoise_iterations=config.hsa_denoise_iterations,
            dropout=config.attention_dropout,
            use_trainable_encoder=config.hsa_trainable_encoder,
        )

        # TT-CER MoE
        self.moe = TTCERMoE(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_experts=config.num_experts,
            top_k=config.top_k,
            router_tt_ranks=config.router_tt_ranks,
            expert_tt_ranks=config.expert_tt_ranks,
            capacity_factor=config.capacity_factor,
            activation="gelu",
            dropout=config.hidden_dropout,
            router_aux_loss_coef=config.router_aux_loss_coef,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Track HSA context for analysis
        self.register_buffer('context_history', torch.zeros(1, config.num_heads, config.hsa_dim // config.num_heads))

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # HSA Attention with residual
        residual = x
        x = self.norm1(x)

        attn_out, hsa_context = self.attention(x, positions, return_context=True)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out

        # TT-CER MoE with residual
        residual = x
        x = self.norm2(x)

        moe_out, moe_aux_loss = self.moe(x, return_aux_loss)
        x = residual + moe_out

        return x, moe_aux_loss, hsa_context


class NSTPModel(nn.Module):
    """
    Full NSTP Model.
    Architecture:
    - Standard embedding (separate from lm_head — no weight tying)
    - Stack of NSTPBlock (HSA + TT-CER MoE)
    - Final LayerNorm
    - LM Head (separate linear, untied from embedding)
    """

    def __init__(self, config: NSTPConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        if config.use_tt_embedding:
            self.embedding = TTEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.d_model,
                tt_ranks=config.embedding_tt_ranks,
            )
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NSTPBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.d_model)
        # LM Head — separate from embedding (no weight tying)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        # Special initialization for embeddings
        if not config.use_tt_embedding:
            nn.init.normal_(self.embedding.weight, std=config.init_std)

        print(f"NSTPModel initialized:")
        print(f"  d_model={config.d_model}, hsa_dim={config.hsa_dim}")
        print(f"  num_layers={config.num_layers}, num_experts={config.num_experts}")
        print(f"  hsa_trainable_encoder={config.hsa_trainable_encoder}")
        print(f"  Total params: {self.num_parameters():,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_aux_losses: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        # Embeddings
        x = self.embedding(input_ids)
        x = self.dropout(x)

        # Track auxiliary losses
        aux_losses = {}
        total_moe_loss = 0.0
        total_tt_ortho_loss = 0.0
        hsa_contexts = []

        # Pass through blocks
        for i, block in enumerate(self.blocks):
            x, moe_aux_loss, hsa_context = block(
                x, positions, return_aux_loss=return_aux_losses
            )

            if moe_aux_loss is not None:
                total_moe_loss += moe_aux_loss

            # TT orthogonality loss
            block_ortho_loss = block.moe.tt_orthogonal_loss()
            total_tt_ortho_loss += block_ortho_loss

            if hsa_context is not None:
                hsa_contexts.append(hsa_context)

        # Final norm
        x = self.norm(x)

        # LM head
        logits = self.lm_head(x)

        # Prepare auxiliary losses
        if return_aux_losses:
            aux_losses = {
                'moe_load_balance': total_moe_loss,
                'tt_orthogonality': total_tt_ortho_loss,
            }

            # Scale by coefficients
            aux_losses['moe_load_balance'] *= self.config.router_aux_loss_coef
            aux_losses['tt_orthogonality'] *= self.config.tt_ortho_loss_coef

        return logits, aux_losses

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        device = input_ids.device
        batch = input_ids.shape[0]

        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

            logits, _ = self.forward(input_ids, positions, return_aux_losses=False)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, -float('inf'))
                next_logits.scatter_(1, top_k_idx, top_k_vals)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs > top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                indices_to_remove = sorted_idx_to_remove.scatter(1, sorted_idx, sorted_idx_to_remove)
                next_logits[indices_to_remove] = -float('inf')

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_compression_stats(self) -> Dict:
        """Get model compression statistics."""
        stats = {
            'total_params': self.num_parameters(),
            'embedding': {},
            'blocks': [],
        }

        if self.config.use_tt_embedding:
            W = self.embedding.reconstruct_weight()
            stats['embedding'] = {
                'type': 'TT',
                'params': sum(c.numel() for c in self.embedding.cores),
                'dense_params': self.config.vocab_size * self.config.d_model,
                'compression': (self.config.vocab_size * self.config.d_model) /
                              sum(c.numel() for c in self.embedding.cores),
            }

        for i, block in enumerate(self.blocks):
            block_stats = block.moe.get_compression_stats()
            block_stats['layer'] = i
            stats['blocks'].append(block_stats)

        total_dense = 0
        total_compressed = stats['total_params']
        for b in stats['blocks']:
            total_dense += b['total_dense']

        stats['total_dense_estimate'] = total_dense
        stats['overall_compression'] = total_dense / total_compressed if total_compressed > 0 else 0

        return stats

    def orthogonalize_all_cores(self):
        """Orthogonalize all TT-cores (stub for Phase 1)."""
        pass


def create_nstp_model(config_dict: Dict[str, Any]) -> NSTPModel:
    """Create NSTPModel from configuration dictionary."""
    config = NSTPConfig(**config_dict)
    return NSTPModel(config)


if __name__ == "__main__":
    print("Testing NSTP Model...")

    config = NSTPConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=4,
        hsa_dim=4096,
        num_experts=4,
        top_k=2,
        d_ff=1024,
        router_tt_ranks=[1, 8, 8, 1],
        expert_tt_ranks=[1, 8, 8, 8, 1],
        embedding_tt_ranks=[1, 8, 8, 1],
        use_tt_embedding=True,
    )

    model = NSTPModel(config)

    x = torch.randint(0, 1000, (2, 128))
    logits, aux_losses = model(x)
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Aux losses: {aux_losses}")

    stats = model.get_compression_stats()
    print(f"\nCompression stats:")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Overall compression: {stats['overall_compression']:.1f}x")

    print("\nGenerating...")
    generated = model.generate(x[:1, :10], max_new_tokens=5)
    print(f"Generated: {generated.shape}")

    print("All tests passed!")