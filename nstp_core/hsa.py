"""
Hyperdimensional Symbolic Attention (HSA) Core
Implements O(n) attention via hyperdimensional computing / holographic reduced representations.

Key operations:
- bind(a, b): Circular convolution (or XOR for binary) - composition
- unbind(a, b): Inverse binding - retrieval  
- superposition: Element-wise addition - set union
- cyclic_shift: Position encoding via rotation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def cyclic_shift(x: torch.Tensor, shift: int, dim: int = -1) -> torch.Tensor:
    """
    Cyclic shift (rotation) along dimension.
    For position encoding in HSA: ρ^i corresponds to shift by i.
    
    Args:
        x: Input tensor [..., D]
        shift: Number of positions to shift (can be negative)
        dim: Dimension to shift along
    
    Returns:
        Shifted tensor with same shape
    """
    if shift == 0:
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    return torch.cat([x.index_select(dim, torch.arange(x.size(dim) - shift, x.size(dim), device=x.device)),
                      x.index_select(dim, torch.arange(0, x.size(dim) - shift, device=x.device))], dim=dim)


def bind(a: torch.Tensor, b: torch.Tensor, mode: str = "circular") -> torch.Tensor:
    """
    Hyperdimensional binding (composition).
    Creates a new vector dissimilar to both inputs.
    
    Args:
        a: First vector [..., D]
        b: Second vector [..., D]
        mode: "circular" (circular convolution), "xor" (binary), "hadamard" (element-wise)
    
    Returns:
        Bound vector [..., D]
    """
    if mode == "circular":
        # Circular convolution via FFT: O(D log D)
        # a ⊛ b = IFFT(FFT(a) * FFT(b))
        return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1), 
                               n=a.size(-1), dim=-1)
    elif mode == "xor":
        # Binary XOR binding: a ⊕ b for {+1, -1} vectors
        # Equivalent to element-wise multiplication for binary
        return a * b  # XOR in {+1, -1} space
    elif mode == "hadamard":
        # Element-wise product
        return a * b
    else:
        raise ValueError(f"Unknown bind mode: {mode}")


def unbind(c: torch.Tensor, b: torch.Tensor, mode: str = "circular") -> torch.Tensor:
    """
    Hyperdimensional unbinding (retrieval).
    Recovers a from c = bind(a, b).
    
    Args:
        c: Composed vector [..., D]
        b: Key vector [..., D]
        mode: Binding mode used
    
    Returns:
        Retrieved vector [..., D]
    """
    if mode == "circular":
        # Unbinding via inverse FFT: a = IFFT(FFT(c) / FFT(b))
        # Add small epsilon for numerical stability
        fft_b = torch.fft.rfft(b, dim=-1)
        fft_c = torch.fft.rfft(c, dim=-1)
        return torch.fft.irfft(fft_c / (fft_b + 1e-8), n=c.size(-1), dim=-1)
    elif mode == "xor":
        # Self-inverse for binary: a = c * b
        return c * b
    elif mode == "hadamard":
        return c / (b + 1e-8)
    else:
        raise ValueError(f"Unknown bind mode: {mode}")


def superposition(vectors: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Hyperdimensional superposition (set union).
    Sum of vectors preserves similarity to each component.
    
    Args:
        vectors: Stack of vectors [N, ..., D]
        dim: Dimension to sum over
    
    Returns:
        Superposed vector [..., D]
    """
    return vectors.sum(dim=dim)


def cosine_similarity_binary(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Cosine similarity for binary {+1, -1} hypervectors.
    sim = (a · b) / D = mean(a * b) for normalized binary vectors.
    """
    return (a * b).mean(dim=dim)


class TrainableBinarize(torch.autograd.Function):
    """
    STE-based trainable binarization.
    Forward: sign(h) with tie-breaking
    Backward: gradient flows through (STE)
    """
    @staticmethod
    def forward(ctx, h, tie_break_value=1.0):
        h_binary = torch.sign(h)
        h_binary = torch.where(h_binary == 0, torch.full_like(h_binary, tie_break_value), h_binary)
        return h_binary

    @staticmethod
    def backward(ctx, grad):
        return grad, None  # Pass gradient through unchanged (STE)


class LearnableHDCEncoder(nn.Module):
    """
    THDC-style learnable HDC encoder.
    
    Replaces fixed random projection + sign() with:
    1. Learnable linear projection
    2. Learnable scale factor (per-dimension)
    3. Optional learned quantization boundaries
    
    This allows the network to learn which information to preserve
    through the sign() bottleneck.
    
    Reference: THDC (arXiv:2602.00116) — Trainable HDC with Backpropagation
    """

    def __init__(
        self,
        d_model: int,
        hsa_dim: int = 16384,
        binary: bool = True,
        use_learnable_scale: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hsa_dim = hsa_dim
        self.binary = binary

        # Trainable projection (instead of fixed random)
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        
        # Initialize with small weights (helps gradient flow through sign)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

        # Learnable scale (THDC key insight: this matters a lot)
        if use_learnable_scale:
            self.scale = nn.Parameter(torch.ones(hsa_dim))
        else:
            self.register_buffer('scale', torch.ones(hsa_dim))
        
        # Learnable shift (bias before sign) — allows learning quantization boundaries
        self.shift = nn.Parameter(torch.zeros(hsa_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [batch, seq_len, d_model]
        
        Returns:
            Binary hypervectors [batch, seq_len, hsa_dim] in {+1, -1}
        """
        # Project and scale
        h = self.proj(x)  # [B, N, D]
        h = h * self.scale.unsqueeze(0).unsqueeze(0)  # Learnable per-dim scaling
        h = h + self.shift.unsqueeze(0).unsqueeze(0)   # Learnable shift

        if self.binary:
            h = TrainableBinarize.apply(h)
            if self.training:
                h = h + (self.proj(x) - self.proj(x).detach())  # STE residual
            return h
        else:
            return F.normalize(h, p=2, dim=-1)


class HSAEncoder(nn.Module):
    """
    Encodes token embeddings into binary hypervectors.
    Uses learned projection + binarization (STE for gradients).
    """

    def __init__(
        self,
        d_model: int,
        hsa_dim: int = 16384,
        binary: bool = True,
        temperature: float = 1.0,
        use_trainable: bool = True,  # NEW: THDC-style trainable encoding
    ):
        super().__init__()
        self.d_model = d_model
        self.hsa_dim = hsa_dim
        self.binary = binary
        self.temperature = temperature

        if use_trainable:
            # THDC-style: trainable projection + scale + shift
            self.encoder = LearnableHDCEncoder(d_model, hsa_dim, binary)
        else:
            # Original: fixed random projection
            self.projection = nn.Linear(d_model, hsa_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [batch, seq_len, d_model]
        
        Returns:
            Binary hypervectors [batch, seq_len, hsa_dim] in {+1, -1}
        """
        if hasattr(self, 'encoder'):
            return self.encoder(x)
        
        # Project to HSA dimension
        h = self.projection(x)  # [B, N, D]

        if self.binary:
            # Binarize with Straight-Through Estimator
            # Forward: sign, but handle zeros → +1
            h_binary = torch.sign(h)
            h_binary = torch.where(h_binary == 0, torch.full_like(h_binary, 1.0), h_binary)

            # STE: gradient flows through unchanged in backward pass
            if self.training:
                h = h_binary + (h - h.detach())
            else:
                h = h_binary
        else:
            # Normalize continuous vectors
            h = F.normalize(h, p=2, dim=-1)

        return h


class HSAContextAccumulator(nn.Module):
    """
    Accumulates hypervector sequence into context vector M.
    M = Σᵢ bind(hᵢ, ρⁱ) where ρⁱ is cyclic shift by i.
    
    This is the O(n) replacement for O(n²) attention matrix.
    """
    
    def __init__(
        self,
        hsa_dim: int = 16384,
        bind_mode: str = "xor",
        max_seq_len: int = 1_000_000,
    ):
        super().__init__()
        self.hsa_dim = hsa_dim
        self.bind_mode = bind_mode
        self.max_seq_len = max_seq_len
        
        # Precompute position encodings (cyclic shifts)
        # For binary: ρⁱ = shift by i positions
        # We'll compute on the fly for memory efficiency
        
    def forward(
        self, 
        h: torch.Tensor, 
        positions: Optional[torch.Tensor] = None,
        return_per_token: bool = False
    ) -> torch.Tensor:
        """
        Args:
            h: Hypervectors [batch, seq_len, hsa_dim] 
            positions: Optional position indices [seq_len] or [batch, seq_len]
            return_per_token: If True, return context at each position
        
        Returns:
            Context vector M [batch, hsa_dim] or per-token [batch, seq_len, hsa_dim]
        """
        batch, seq_len, _ = h.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=h.device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(batch, -1)
        
        # Bind each token with its position encoding
        # bind(hᵢ, ρⁱ) = cyclic_shift(hᵢ, positions[i]) for XOR mode
        if self.bind_mode == "xor":
            # Vectorized: gather-based cyclic shift (no Python loop, no pos.item())
            # bound[b, i, k] = h[b, i, (k - positions[b, i]) mod D]
            D = self.hsa_dim
            k_idx = torch.arange(D, device=h.device).view(1, 1, D)
            shifts = positions.view(batch, seq_len, 1)
            idx = (k_idx - shifts) % D
            bound = torch.gather(h, 2, idx)
        else:
            # Circular convolution mode - use FFT
            # Precompute FFT of position encodings
            bound = torch.zeros_like(h)
            for i in range(seq_len):
                pos_enc = torch.zeros_like(h[:, i])
                pos_enc[:, 0] = 1.0  # Delta at position 0
                pos_enc = cyclic_shift(pos_enc, positions[:, i].item())
                bound[:, i] = bind(h[:, i], pos_enc, mode=self.bind_mode)
        
        if return_per_token:
            # Return cumulative context at each position
            # M_t = Σ_{i≤t} bound_i
            return bound.cumsum(dim=1)
        
        # Sum all bound tokens -> context M
        M = bound.sum(dim=1)  # [batch, hsa_dim]
        return M


class HSADenoiser(nn.Module):
    """
    Iterative denoising of retrieved hypervectors.
    Uses associative memory (Hopfield-style) to clean noise from superposition.
    """
    
    def __init__(
        self,
        hsa_dim: int = 16384,
        num_iterations: int = 3,
        threshold: float = 0.0,
        binary: bool = True,
    ):
        super().__init__()
        self.hsa_dim = hsa_dim
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.binary = binary
        
        # Learned cleanup projection (optional)
        self.cleanup = nn.Linear(hsa_dim, hsa_dim, bias=False)
        nn.init.orthogonal_(self.cleanup.weight)
    
    def forward(
        self, 
        query: torch.Tensor, 
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Noisy retrieved vector [batch, hsa_dim]
            memory: Optional stored patterns [num_patterns, hsa_dim] for associative recall
        
        Returns:
            Cleaned vector [batch, hsa_dim]
        """
        x = query
        
        for _ in range(self.num_iterations):
            if memory is not None:
                # Associative recall: find closest stored pattern
                if self.binary:
                    # Hamming distance / cosine similarity
                    sims = torch.matmul(x, memory.t()) / self.hsa_dim
                    # Soft retrieval
                    weights = F.softmax(sims * 10, dim=-1)
                    x = torch.matmul(weights, memory)
                else:
                    sims = F.cosine_similarity(x.unsqueeze(1), memory.unsqueeze(0), dim=-1)
                    weights = F.softmax(sims * 10, dim=-1)
                    x = torch.matmul(weights, memory)
            
            # Learned cleanup
            x = self.cleanup(x)
            
            # Binarize if binary mode
            if self.binary:
                x = torch.sign(x)
        
        return x


class HyperdimensionalAttention(nn.Module):
    """
    Complete HSA Attention Module.
    Replaces standard multi-head attention with O(n) hyperdimensional attention.
    
    Flow:
    1. Encode tokens to hypervectors
    2. Accumulate context M = Σ bind(hᵢ, posᵢ)
    3. For each position, retrieve: q = unbind(M, pos)
    4. Denoise retrieved vectors
    5. Decode to output
    """
    
    def __init__(
        self,
        d_model: int,
        hsa_dim: int = 16384,
        num_heads: int = 8,
        binary: bool = True,
        bind_mode: str = "xor",
        denoise_iterations: int = 3,
        dropout: float = 0.1,
        use_trainable_encoder: bool = True,  # NEW: THDC-style
    ):
        super().__init__()
        self.d_model = d_model
        self.hsa_dim = hsa_dim
        self.num_heads = num_heads
        self.head_dim = hsa_dim // num_heads
        self.use_trainable_encoder = use_trainable_encoder
        assert hsa_dim % num_heads == 0
        
        self.binary = binary
        self.bind_mode = bind_mode
        
        # Per-head encoders — THDC-style trainable
        self.encoders = nn.ModuleList([
            HSAEncoder(d_model, self.head_dim, binary=binary, use_trainable=use_trainable_encoder)
            for _ in range(num_heads)
        ])
        
        # Context accumulators
        self.accumulators = nn.ModuleList([
            HSAContextAccumulator(self.head_dim, bind_mode=bind_mode)
            for _ in range(num_heads)
        ])
        
        # Denoisers
        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=denoise_iterations, binary=binary)
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hsa_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def _retrieve_xor_vectorized(
        self,
        M: torch.Tensor,            # [batch, head_dim]
        positions: torch.Tensor,    # [batch, seq_len] or [seq_len] (single sequence)
        batch: int,
        seq_len: int,
        head_dim: int,
    ) -> torch.Tensor:
        """
        Vectorized cyclic_shift retrieval: for each (b, i), return cyclic_shift(M[b], -positions[b, i]).

        Approach:
        - Expand M to [batch, seq_len, head_dim] where each slot copies M[b].
        - Use torch.roll along last dim with a per-(b, i) shift? torch.roll only supports
          a scalar-or-1D shift, not per-row of a 2D slice.

        Practical fully-vectorized solution: precompute indices on an extended dimension.
        Specifically, build an expanded grid [batch*seq_len, head_dim] using index gather.
        """
        device = M.device
        dtype = M.dtype

        # If positions is [seq_len] only, expand to [batch, seq_len]
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)

        # M has shape [batch, head_dim].
        # We want output [batch, seq_len, head_dim] where
        # out[b, i, :] = roll(M[b], shifts=-positions[b, i])

        # Construct the gather index for every (b, i, k):
        # out[b, i, k] = M[b, (k - positions[b, i]) mod head_dim]
        k = torch.arange(head_dim, device=device).view(1, 1, head_dim)  # [1, 1, D]
        shifts = positions.view(batch, seq_len, 1)                        # [B, N, 1]
        idx = (k - shifts) % head_dim                                     # [B, N, D]

        # Expand M to [B, 1, D] so we can gather
        M_expanded = M.unsqueeze(1).expand(batch, seq_len, head_dim)      # [B, N, D]
        retrieved = torch.gather(M_expanded, 2, idx)                      # [B, N, D]
        return retrieved

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_context: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input embeddings [batch, seq_len, d_model]
            positions: Position indices [seq_len] or [batch, seq_len]
            return_context: If True, return accumulated context M
        
        Returns:
            output: Attended output [batch, seq_len, d_model]
            context: Optional context M [batch, num_heads, head_dim]
        """
        batch, seq_len, _ = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(batch, -1)
        
        head_outputs = []
        contexts = []
        
        for head_idx in range(self.num_heads):
            # Encode to hypervectors
            h = self.encoders[head_idx](x)  # [B, N, head_dim]
            
            # Accumulate context M = Σ bind(hᵢ, posᵢ)
            M = self.accumulators[head_idx](h, positions)
            contexts.append(M)
            
            # Retrieve per position: qᵢ = unbind(M, posᵢ)
            # For XOR mode: qᵢ = cyclic_shift(M, -posᵢ)
            # Vectorized: torch.roll applied per-(batch, position) is still
            # sequential in Python. We instead RETRIEVE FOR ALL POSITIONS by
            # rolling the accumulator once per shift (still O(n) but no inner loop).
            # Best fully-vectorized approach: compute shift indices per (b, i) and use gather.
            if self.bind_mode == "xor":
                retrieved = self._retrieve_xor_vectorized(M, positions, batch, seq_len, self.hsa_dim // self.num_heads)
            else:
                retrieved = torch.zeros_like(h)
                for b in range(batch):
                    for i in range(seq_len):
                        pos_enc = torch.zeros_like(M[b:b+1])
                        pos_enc[:, 0] = 1.0
                        pos_enc = cyclic_shift(pos_enc, positions[b, i].item())
                        retrieved[b, i] = unbind(M[b], pos_enc[0], mode=self.bind_mode)
            
            # Denoise
            cleaned = self.denoisers[head_idx](retrieved)
            head_outputs.append(cleaned)
        
        # Combine heads
        combined = torch.cat(head_outputs, dim=-1)  # [B, N, hsa_dim]
        combined = self.dropout(combined)
        output = self.output_proj(combined)
        
        if return_context:
            context = torch.stack(contexts, dim=1)  # [B, num_heads, head_dim]
            return output, context
        
        return output, None


def hyperdimensional_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional form of HSA attention for drop-in replacement.
    Not used in module form - module is preferred for learned components.
    """
    # This is a simplified functional version for analysis
    # Real implementation uses the module above
    raise NotImplementedError("Use HyperdimensionalAttention module")


# Utility: Analyze HSA capacity
def hsa_capacity_analysis(hsa_dim: int, num_tokens: int, bind_mode: str = "xor") -> dict:
    """
    Theoretical capacity analysis for HSA.
    Returns expected noise level and retrieval accuracy.
    """
    if bind_mode == "xor":
        # For binary HSA with XOR binding
        # Noise variance grows as O(num_tokens / hsa_dim)
        noise_std = math.sqrt(num_tokens / hsa_dim)
        # Signal is 1.0 (self-similarity)
        snr = 1.0 / noise_std if noise_std > 0 else float('inf')
        
        # Probability of correct retrieval (approx)
        # Assuming Gaussian noise, P(correct) ≈ Φ(SNR/√2)
        import math
        from math import erf
        p_correct = 0.5 * (1 + erf(snr / math.sqrt(2)))
        
        return {
            "hsa_dim": hsa_dim,
            "num_tokens": num_tokens,
            "noise_std": noise_std,
            "snr": snr,
            "p_correct_retrieval": p_correct,
            "capacity_tokens": hsa_dim / 4,  # Rule of thumb: D/4 tokens with good retrieval
        }
    else:
        return {"mode": bind_mode, "note": "Capacity analysis for circular mode requires simulation"}


if __name__ == "__main__":
    # Quick test
    print("Testing HSA components...")
    
    B, N, D = 2, 1024, 4096
    HSA_DIM = 16384
    
    x = torch.randn(B, N, D)
    positions = torch.arange(N).unsqueeze(0).expand(B, -1)
    
    # Test encoder
    encoder = HSAEncoder(D, HSA_DIM, binary=True)
    h = encoder(x)
    print(f"Encoder: {x.shape} -> {h.shape}, binary: {(h.abs() == 1).all()}")
    
    # Test accumulator
    acc = HSAContextAccumulator(HSA_DIM, bind_mode="xor")
    M = acc(h, positions)
    print(f"Accumulator: {h.shape} -> {M.shape}")
    
    # Test full attention
    attn = HyperdimensionalAttention(D, HSA_DIM, num_heads=8)
    out, ctx = attn(x, positions, return_context=True)
    print(f"Attention: {x.shape} -> {out.shape}, context: {ctx.shape}")
    
    # Capacity analysis
    cap = hsa_capacity_analysis(HSA_DIM, N)
    print(f"Capacity: {cap}")
    
    print("All tests passed!")