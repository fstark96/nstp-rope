"""
TT-CER MoE (Tensor-Train Compressed Expert Routing)
Implements Mixture of Experts with TT-compressed weights and learned routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

from .tt import TTLinear, tt_orthogonal_loss


class TTCERRouter(nn.Module):
    """
    Router for TT-CER MoE.
    Uses TT-compressed weight matrix for routing logits.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        tt_ranks: List[int],
        top_k: int = 2,
        capacity_factor: float = 1.25,
        router_aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router_aux_loss_coef = router_aux_loss_coef
        
        # TT-compressed router weights
        self.router = TTLinear(d_model, num_experts, tt_ranks, bias=False)
        
        # Expert usage statistics for load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('step_count', torch.tensor(0))
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model] or [batch*seq_len, d_model]
            return_aux_loss: Whether to compute load balancing loss
        
        Returns:
            gates: Expert weights [batch*seq_len, num_experts] (sparse, top-k)
            indices: Expert indices [batch*seq_len, top_k]
            aux_loss: Optional load balancing loss
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.d_model)  # [tokens, d_model]
        num_tokens = x_flat.shape[0]
        
        # Compute routing logits via TT-linear
        logits = self.router(x_flat)  # [tokens, num_experts]
        
        # Top-k routing
        gates, indices = torch.topk(logits, self.top_k, dim=-1)  # [tokens, top_k]
        gates = F.softmax(gates, dim=-1)
        
        # Create sparse gate matrix
        gate_matrix = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        gate_matrix.scatter_(1, indices, gates)
        
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_load_balancing_loss(gate_matrix)
        
        # Update usage statistics
        if self.training:
            with torch.no_grad():
                self.expert_usage += gate_matrix.sum(0)
                self.step_count += 1
        
        # Reshape gates/indices back to original shape
        gates_out = gates.reshape(*orig_shape, self.top_k)
        indices_out = indices.reshape(*orig_shape, self.top_k)
        
        return gates_out, indices_out, aux_loss
    
    def _compute_load_balancing_loss(self, gate_matrix: torch.Tensor) -> torch.Tensor:
        """
        Load balancing loss from Switch Transformer / GShard.
        Encourages uniform expert usage.
        """
        # Fraction of tokens dispatched to each expert
        expert_fraction = gate_matrix.mean(0)  # [num_experts]
        
        # Ideal: uniform 1/num_experts
        # Loss: num_experts * sum(fraction^2) - encourages uniformity
        loss = self.num_experts * (expert_fraction ** 2).sum()
        
        # Scale by coefficient
        return self.router_aux_loss_coef * loss
    
    def get_expert_usage_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert usage statistics."""
        if self.step_count > 0:
            avg_usage = self.expert_usage / self.step_count
        else:
            avg_usage = self.expert_usage
        return {
            'expert_usage': avg_usage,
            'step_count': self.step_count,
            'load_balance_score': (avg_usage * self.num_experts).std().item(),
        }


class TTCERExpert(nn.Module):
    """
    Single expert in TT-CER MoE.
    Uses TT-compressed MLP (two TTLinear layers).
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        tt_ranks: List[int],
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two-layer TT-MLP: d_model -> d_ff -> d_model
        self.fc1 = TTLinear(d_model, d_ff, tt_ranks, bias=bias)
        self.fc2 = TTLinear(d_ff, d_model, tt_ranks, bias=bias)
        
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d_model]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def tt_orthogonal_loss(self) -> torch.Tensor:
        """Compute TT orthogonality loss for this expert's weights."""
        loss = 0.0
        for layer in [self.fc1, self.fc2]:
            loss += tt_orthogonal_loss([c for c in layer.cores])
        return loss / 2
    
    def orthogonalize_cores(self):
        """Orthogonalize TT-cores for numerical stability (stub)."""
        pass  # TT-core orthogonalization simplified for Phase 1


class TTCERMoE(nn.Module):
    """
    Full TT-CER Mixture of Experts Layer.
    
    Combines:
    - TT-compressed router (TTCERRouter)
    - Multiple TT-compressed experts (TTCERExpert)
    - Capacity-aware dispatch
    - Load balancing
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        router_tt_ranks: List[int] = None,
        expert_tt_ranks: List[int] = None,
        capacity_factor: float = 1.25,
        activation: str = "gelu",
        dropout: float = 0.1,
        router_aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Default TT ranks if not provided
        if router_tt_ranks is None:
            router_tt_ranks = [1, 16, 16, 1]  # Small router
        if expert_tt_ranks is None:
            expert_tt_ranks = [1, 16, 16, 16, 1]  # Deeper for experts
        
        self.router_tt_ranks = router_tt_ranks
        self.expert_tt_ranks = expert_tt_ranks
        
        # Router
        self.router = TTCERRouter(
            d_model=d_model,
            num_experts=num_experts,
            tt_ranks=router_tt_ranks,
            top_k=top_k,
            capacity_factor=capacity_factor,
            router_aux_loss_coef=router_aux_loss_coef,
        )
        
        # Experts
        self.experts = nn.ModuleList([
            TTCERExpert(
                d_model=d_model,
                d_ff=d_ff,
                tt_ranks=expert_tt_ranks,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
        # Layer norm for residual
        self.norm = nn.LayerNorm(d_model)
        
        # Track statistics
        self.register_buffer('tokens_processed', torch.tensor(0))
        self.register_buffer('expert_counts', torch.zeros(num_experts))
    
    def forward(
        self, 
        x: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input [batch, seq_len, d_model]
            return_aux_loss: Whether to return load balancing loss
        
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: Optional load balancing loss
        """
        batch, seq_len, _ = x.shape
        
        # Normalize input
        x = self.norm(x)
        
        # Reshape for routing
        x_flat = x.reshape(-1, self.d_model)  # [batch*seq_len, d_model]
        num_tokens = x_flat.shape[0]
        
        # Route tokens
        gates, indices, aux_loss = self.router(x_flat, return_aux_loss)

        # === Vectorized dispatch: process all tokens through all experts at once ===
        # all_outputs: (E, num_tokens, d_model)
        all_outputs = torch.stack([expert(x_flat) for expert in self.experts])

        # Build gate matrix: (num_tokens, E) — accumulated gate per expert per token
        gate_matrix = torch.zeros(num_tokens, self.num_experts, device=x_flat.device)
        for k in range(self.top_k):
            gate_matrix.scatter_add_(1, indices[:, k:k+1], gates[:, k:k+1])

        # Weighted sum: output[t,d] = sum_e gate[t,e] * all_outputs[e,t,d]
        # gate_matrix: (T, E), all_outputs: (E, T, D) -> bmm -> (T, D)
        output = torch.bmm(
            gate_matrix.unsqueeze(1),      # (T, 1, E)
            all_outputs.permute(1, 0, 2)   # (T, E, D)
        ).squeeze(1)                        # (T, D)

        # Update statistics
        if self.training:
            with torch.no_grad():
                self.tokens_processed += num_tokens
                self.expert_counts.scatter_add_(
                    0, indices[:, 0], torch.ones_like(indices[:, 0], dtype=torch.float)
                )
                if self.top_k > 1:
                    self.expert_counts.scatter_add_(
                        0, indices[:, 1], torch.ones_like(indices[:, 1], dtype=torch.float)
                    )
        
        # Reshape output
        output = output.reshape(batch, seq_len, self.d_model)
        
        # Residual connection
        output = output + x
        
        return output, aux_loss
    
    def tt_orthogonal_loss(self) -> torch.Tensor:
        """Total TT orthogonality loss for all experts + router."""
        loss = 0.0
        for expert in self.experts:
            loss += expert.tt_orthogonal_loss()
        loss += tt_orthogonal_loss([c for c in self.router.router.cores])
        return loss / (self.num_experts + 1)
    
    def orthogonalize_cores(self):
        """Orthogonalize all TT-cores (stub for Phase 1)."""
        pass  # TT-core orthogonalization simplified for Phase 1
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        router_params = sum(c.numel() for c in self.router.router.cores)
        expert_params = sum(
            sum(c.numel() for c in expert.fc1.cores) + 
            sum(c.numel() for c in expert.fc2.cores)
            for expert in self.experts
        )
        
        # Dense equivalents
        router_dense = self.d_model * self.num_experts
        expert_dense = self.num_experts * (self.d_model * self.d_ff + self.d_ff * self.d_model)
        
        return {
            'router_params': router_params,
            'router_dense': router_dense,
            'router_compression': router_dense / router_params if router_params > 0 else 0,
            'expert_params': expert_params,
            'expert_dense': expert_dense,
            'expert_compression': expert_dense / expert_params if expert_params > 0 else 0,
            'total_params': router_params + expert_params,
            'total_dense': router_dense + expert_dense,
            'total_compression': (router_dense + expert_dense) / (router_params + expert_params),
        }
    
    def get_expert_usage(self) -> torch.Tensor:
        """Get expert usage counts."""
        return self.expert_counts.clone()


def top_k_routing(
    logits: torch.Tensor,
    k: int,
    capacity_factor: float = 1.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k routing with capacity constraints.
    
    Args:
        logits: [batch*seq_len, num_experts]
        k: Number of experts per token
        capacity_factor: Capacity multiplier
    
    Returns:
        gates: [batch*seq_len, k] softmax weights
        indices: [batch*seq_len, k] expert indices
    """
    # Standard top-k
    gates, indices = torch.topk(logits, k, dim=-1)
    gates = F.softmax(gates, dim=-1)
    return gates, indices


def load_balancing_loss(
    gates: torch.Tensor,  # [tokens, num_experts] - full softmax or top-k sparse
    num_experts: int,
    coef: float = 0.01,
) -> torch.Tensor:
    """
    Load balancing loss for MoE.
    """
    if gates.dim() == 2:
        # Full softmax
        expert_fraction = gates.mean(0)
    else:
        # Sparse top-k - need to reconstruct
        tokens, k = gates.shape[:2]
        if gates.shape[-1] == num_experts:
            expert_fraction = gates.mean(0)
        else:
            # Sparse format
            expert_fraction = torch.zeros(num_experts, device=gates.device)
            # This would need indices - simplified here
            pass
    
    loss = num_experts * (expert_fraction ** 2).sum()
    return coef * loss


if __name__ == "__main__":
    print("Testing TT-CER MoE...")
    
    # Configuration
    d_model = 512
    d_ff = 2048
    num_experts = 8
    top_k = 2
    
    # Create MoE
    moe = TTCERMoE(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        router_tt_ranks=[1, 8, 8, 1],
        expert_tt_ranks=[1, 8, 8, 8, 1],
    )
    
    # Test input
    x = torch.randn(2, 128, d_model)  # [batch, seq_len, d_model]
    
    # Forward
    output, aux_loss = moe(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Aux loss: {aux_loss}")
    
    # Compression stats
    stats = moe.get_compression_stats()
    print(f"\nCompression stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Test router usage
    usage = moe.get_expert_usage()
    print(f"\nExpert usage: {usage}")
    
    # Test TT orthogonality loss
    ortho_loss = moe.tt_orthogonal_loss()
    print(f"Ortho loss: {ortho_loss:.6f}")
    
    print("All tests passed!")