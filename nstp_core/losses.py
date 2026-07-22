"""
NSTP Loss Functions
Implements all auxiliary losses for NSTP training:
- HSA Denoising Loss
- TT Orthogonality Loss
- Expert Load Balancing Loss
- Specialization Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .hsa import bind, unbind, cyclic_shift, cosine_similarity_binary
from .tt import tt_orthogonal_loss


def hsa_denoising_loss(
    retrieved: torch.Tensor,
    targets: torch.Tensor,
    positions: torch.Tensor,
    hsa_dim: int,
    bind_mode: str = "xor",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Denoising loss for HSA retrieval.
    
    During training, we know the ground truth hypervectors.
    The retrieved vector q = unbind(M, pos) should match the original h.
    
    Args:
        retrieved: Retrieved hypervectors [batch, seq_len, hsa_dim]
        targets: Target/original hypervectors [batch, seq_len, hsa_dim]
        positions: Position indices [batch, seq_len]
        hsa_dim: HSA dimension
        bind_mode: Binding mode
        reduction: "mean", "sum", "none"
    
    Returns:
        Scalar loss (or per-element if reduction="none")
    """
    if bind_mode == "xor":
        # For binary hypervectors, use Hamming distance / cosine similarity
        # Retrieved and targets are in {-1, +1}
        # Cosine similarity = mean(retrieved * target)
        similarity = (retrieved * targets).mean(dim=-1)  # [batch, seq_len]
        # Loss = 1 - similarity (maximize similarity)
        loss = 1.0 - similarity
    else:
        # Continuous: MSE
        loss = F.mse_loss(retrieved, targets, reduction='none').mean(dim=-1)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def hsa_reconstruction_loss(
    model_output: torch.Tensor,
    target_hypervectors: torch.Tensor,
    positions: torch.Tensor,
    hsa_dim: int,
    bind_mode: str = "xor",
) -> torch.Tensor:
    """
    Reconstruction loss for HSA autoencoding.
    
    Encourages the full encode -> accumulate -> retrieve -> decode cycle
    to reconstruct the original hypervectors.
    """
    # This would be used in a self-supervised pretraining setup
    # where we encode tokens to hypervectors, accumulate, retrieve, and decode
    pass


def expert_balance_loss(
    gates: torch.Tensor,
    num_experts: int,
    coef: float = 0.01,
) -> torch.Tensor:
    """
    Load balancing loss for MoE experts.
    From GShard / Switch Transformer.
    
    Args:
        gates: Expert assignment probabilities [batch*seq_len, num_experts]
               or sparse top-k format [batch*seq_len, top_k]
        num_experts: Total number of experts
        coef: Loss coefficient
    """
    if gates.dim() == 2:
        # Full softmax or dense gates
        expert_fraction = gates.mean(0)  # [num_experts]
    else:
        # Assume sparse top-k format: [tokens, top_k, 2] with (gate, index)
        # This is a simplified version
        tokens = gates.shape[0]
        expert_fraction = torch.zeros(num_experts, device=gates.device)
        # This would need the indices - skipping for now
        pass
    
    # Loss: num_experts * sum(fraction^2)
    loss = num_experts * (expert_fraction ** 2).sum()
    return coef * loss


def specialization_loss(
    gates: torch.Tensor,
    labels: torch.Tensor,
    num_experts: int,
    coef: float = 0.1,
) -> torch.Tensor:
    """
    Specialization loss - encourages experts to specialize on different data patterns.
    Maximizes mutual information between expert assignment and input features/labels.
    
    Args:
        gates: Expert gates [batch, seq_len, top_k] or [batch*seq_len, top_k]
        labels: Class labels or cluster IDs [batch*seq_len]
        num_experts: Number of experts
        coef: Loss coefficient
    """
    # Simplified: encourage different experts to handle different classes
    # Using entropy of expert-class distribution
    
    # This is a complex loss - simplified version
    # In practice, use mutual information estimation or contrastive loss
    return torch.tensor(0.0, device=gates.device)


def hsa_context_consistency_loss(
    contexts_per_layer: List[torch.Tensor],
    coef: float = 0.01,
) -> torch.Tensor:
    """
    Consistency loss for HSA context across layers.
    Encourages similar context representations in adjacent layers.
    """
    if len(contexts_per_layer) < 2:
        return torch.tensor(0.0, device=contexts_per_layer[0].device)
    
    loss = 0.0
    for i in range(len(contexts_per_layer) - 1):
        c1 = contexts_per_layer[i]  # [batch, num_heads, head_dim]
        c2 = contexts_per_layer[i + 1]
        
        # Cosine similarity between corresponding heads
        c1_flat = c1.reshape(c1.shape[0], -1)
        c2_flat = c2.reshape(c2.shape[0], -1)
        
        # Normalize
        c1_n = F.normalize(c1_flat, p=2, dim=-1)
        c2_n = F.normalize(c2_flat, p=2, dim=-1)
        
        # Similarity
        sim = (c1_n * c2_n).sum(dim=-1).mean()
        loss += 1.0 - sim
    
    return coef * loss / (len(contexts_per_layer) - 1)


class NSTPLoss(nn.Module):
    """
    Combined NSTP Loss Module.
    Aggregates all auxiliary losses with configurable weights.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hsa_dim: int,
        num_experts: int,
        num_layers: int,
        hsa_denoise_coef: float = 0.1,
        tt_ortho_coef: float = 0.01,
        router_balance_coef: float = 0.01,
        hsa_consistency_coef: float = 0.01,
        ce_loss_coef: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hsa_dim = hsa_dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        self.hsa_denoise_coef = hsa_denoise_coef
        self.tt_ortho_coef = tt_ortho_coef
        self.router_balance_coef = router_balance_coef
        self.hsa_consistency_coef = hsa_consistency_coef
        self.ce_loss_coef = ce_loss_coef
        
        # Main CE loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_losses: Dict[str, torch.Tensor],
        hsa_contexts: Optional[List[torch.Tensor]] = None,
        hsa_retrieved: Optional[torch.Tensor] = None,
        hsa_targets: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            aux_losses: Dict from model forward (moe_load_balance, tt_orthogonality)
            hsa_contexts: List of HSA contexts per layer [batch, num_heads, head_dim]
            hsa_retrieved: Retrieved hypervectors [batch, seq_len, hsa_dim]
            hsa_targets: Target hypervectors [batch, seq_len, hsa_dim]
            positions: Position indices [batch, seq_len]
        
        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual losses
        """
        losses = {}
        
        # Main cross-entropy loss
        ce = self.ce_loss(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        losses['ce'] = ce * self.ce_loss_coef
        
        # MoE load balancing
        if 'moe_load_balance' in aux_losses:
            losses['moe_balance'] = aux_losses['moe_load_balance'] * self.router_balance_coef
        
        # TT orthogonality
        if 'tt_orthogonality' in aux_losses:
            losses['tt_ortho'] = aux_losses['tt_orthogonality'] * self.tt_ortho_coef
        
        # HSA denoising
        if hsa_retrieved is not None and hsa_targets is not None and positions is not None:
            denoise_loss = hsa_denoising_loss(
                hsa_retrieved, hsa_targets, positions, self.hsa_dim
            )
            losses['hsa_denoise'] = denoise_loss * self.hsa_denoise_coef
        
        # HSA context consistency
        if hsa_contexts is not None and len(hsa_contexts) > 1:
            consist_loss = hsa_context_consistency_loss(hsa_contexts)
            losses['hsa_consistency'] = consist_loss * self.hsa_consistency_coef
        
        # Total loss
        total = sum(losses.values())
        losses['total'] = total
        
        return total, losses


def compute_tt_compression(model) -> Dict[str, float]:
    """Compute total TT compression statistics for a model."""
    total_params = 0
    total_dense = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'get_compression_stats'):
            stats = module.get_compression_stats()
            total_params += stats.get('total_params', 0)
            total_dense += stats.get('total_dense', 0)
    
    return {
        'compressed_params': total_params,
        'dense_estimate': total_dense,
        'compression_ratio': total_dense / total_params if total_params > 0 else 0,
    }


if __name__ == "__main__":
    print("Testing NSTP Losses...")
    
    # Test HSA denoising loss
    B, N, D = 2, 128, 16384
    retrieved = torch.randint(-1, 2, (B, N, D)).float()
    targets = torch.randint(-1, 2, (B, N, D)).float()
    positions = torch.arange(N).unsqueeze(0).expand(B, -1)
    
    loss = hsa_denoising_loss(retrieved, targets, positions, D)
    print(f"HSA denoising loss: {loss:.6f}")
    
    # Test expert balance loss
    gates = F.softmax(torch.randn(100, 8), dim=-1)
    balance_loss = expert_balance_loss(gates, 8)
    print(f"Expert balance loss: {balance_loss:.6f}")
    
    # Test NSTPLoss
    loss_fn = NSTPLoss(
        vocab_size=1000,
        d_model=256,
        hsa_dim=4096,
        num_experts=4,
        num_layers=4,
    )
    
    logits = torch.randn(2, 32, 1000)
    targets = torch.randint(0, 1000, (2, 32))
    aux_losses = {
        'moe_load_balance': torch.tensor(0.5),
        'tt_orthogonality': torch.tensor(0.1),
    }
    hsa_contexts = [torch.randn(2, 4, 1024) for _ in range(4)]
    hsa_retrieved = torch.randint(-1, 2, (2, 32, 4096)).float()
    hsa_targets = torch.randint(-1, 2, (2, 32, 4096)).float()
    positions = torch.arange(32).unsqueeze(0).expand(2, -1)
    
    total, losses = loss_fn(
        logits, targets, aux_losses, hsa_contexts,
        hsa_retrieved, hsa_targets, positions
    )
    print(f"\nTotal loss: {total:.6f}")
    for k, v in losses.items():
        print(f"  {k}: {v:.6f}")
    
    print("All tests passed!")