"""
Tensor-Train (TT) Utilities for NSTP
Phase 1: Uses reconstructed weights (TT-cores for training, dense for forward)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


def _factorize(dim: int, n: int) -> List[int]:
    """Factor dim into n factors whose product equals dim."""
    if n == 1:
        return [dim]
    primes = []
    n_val = dim
    d = 2
    while d * d <= n_val:
        while n_val % d == 0:
            primes.append(d)
            n_val //= d
        d += 1
    if n_val > 1:
        primes.append(n_val)
    buckets = [[] for _ in range(n)]
    for p in sorted(primes, reverse=True):
        idx = min(range(n), key=lambda i: math.prod(buckets[i]) if buckets[i] else 1)
        buckets[idx].append(p)
    return [math.prod(b) if b else 1 for b in buckets]


def tt_decompose(matrix: torch.Tensor, tt_ranks: List[int],
                 shapes: Optional[List[Tuple[int, int]]] = None) -> List[torch.Tensor]:
    """Decompose (M, N) matrix into TT-cores via sequential SVD."""
    M, N = matrix.shape
    k = len(tt_ranks) - 1
    assert tt_ranks[0] == 1 and tt_ranks[-1] == 1

    if shapes is None:
        m_shapes = _factorize(M, k)
        n_shapes = _factorize(N, k)
    else:
        m_shapes = [s[0] for s in shapes]
        n_shapes = [s[1] for s in shapes]

    assert math.prod(m_shapes) == M and math.prod(n_shapes) == N

    cores = []
    mat = matrix.float().reshape(M, N)
    r_prev = 1

    for i in range(k):
        r_next = tt_ranks[i + 1]
        mi, ni = m_shapes[i], n_shapes[i]
        mat_2d = mat.reshape(r_prev * mi * ni, -1)

        U, S, Vh = torch.linalg.svd(mat_2d, full_matrices=False)
        U = U[:, :r_next]
        S = S[:r_next]
        Vh = Vh[:r_next, :]

        core = U.reshape(r_prev, mi, ni, r_next)
        cores.append(core)

        if i < k - 1:
            mat = (torch.diag(S) @ Vh).reshape(r_next, -1)
            r_prev = r_next

    return cores


def reconstruct_tt(cores: List[torch.Tensor]) -> torch.Tensor:
    """Reconstruct (M, N) from TT-cores via iterative contraction."""
    k = len(cores)
    # Start: (m0, n0, r1)
    result = cores[0].squeeze(0)

    for i in range(1, k):
        mi, ni, ri1 = cores[i].shape[1], cores[i].shape[2], cores[i].shape[3]
        ri = cores[i].shape[0]

        if result.dim() == 2:
            # result is (prev_product, ri)
            result = (result @ cores[i].reshape(ri, -1))
            if i < k - 1:
                result = result.reshape(result.shape[0], mi, ni, ri1)
            else:
                result = result.reshape(-1)
        else:
            prev = math.prod(result.shape[:-1])
            result = result.reshape(prev, ri) @ cores[i].reshape(ri, -1)
            if i < k - 1:
                result = result.reshape(prev, mi, ni, ri1)

    M = math.prod(c.shape[1] for c in cores)
    N = math.prod(c.shape[2] for c in cores)
    return result.reshape(M, N)


def tt_orthogonal_loss(cores: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for core in cores:
        r, m, n, r2 = core.shape
        G = core.reshape(r * m, n * r2)
        GGT = G @ G.t()
        I = torch.eye(GGT.shape[0], device=GGT.device, dtype=GGT.dtype)
        loss += (GGT - I).pow(2).mean()
    return loss / max(len(cores), 1)


class TTLinear(nn.Module):
    """
    Linear layer with TT-structured weight.
    Forward: y = x @ W^T + bias where W is stored as TT-cores.
    Uses reconstructed weight for forward (Phase 1 optimization).
    """
    def __init__(self, in_features: int, out_features: int, tt_ranks: List[int],
                 tt_shapes: Optional[List[Tuple[int, int]]] = None, bias: bool = True):
        super().__init__()
        k = len(tt_ranks) - 1
        self.tt_ranks = tt_ranks
        self.in_features = in_features
        self.out_features = out_features

        if tt_shapes is None:
            self.tt_shapes = list(zip(_factorize(out_features, k), _factorize(in_features, k)))
        else:
            self.tt_shapes = tt_shapes

        self.cores = nn.ParameterList()
        for i, (of, inf) in enumerate(self.tt_shapes):
            rp, rn = tt_ranks[i], tt_ranks[i + 1]
            self.cores.append(nn.Parameter(torch.randn(rp, of, inf, rn) * 0.02))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def weight_reconstructed(self) -> torch.Tensor:
        return reconstruct_tt([c for c in self.cores])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_reconstructed
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def num_params(self) -> int:
        return sum(c.numel() for c in self.cores) + (self.bias.numel() if self.bias is not None else 0)


class TTEmbedding(nn.Module):
    """Embedding with TT-structured weight."""
    def __init__(self, num_embeddings: int, embedding_dim: int, tt_ranks: List[int], **kwargs):
        super().__init__()
        k = len(tt_ranks) - 1
        self.tt_shapes = list(zip(_factorize(num_embeddings, k), _factorize(embedding_dim, k)))
        self.cores = nn.ParameterList()
        for i, (vf, ef) in enumerate(self.tt_shapes):
            rp, rn = tt_ranks[i], tt_ranks[i + 1]
            self.cores.append(nn.Parameter(torch.randn(rp, vf, ef, rn) * 0.02))

    def reconstruct_weight(self) -> torch.Tensor:
        return reconstruct_tt([c for c in self.cores])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        W = self.reconstruct_weight()
        return F.embedding(indices, W)


def linear_to_ttlinear(linear: nn.Linear, tt_ranks: List[int]) -> TTLinear:
    tt = TTLinear(linear.in_features, linear.out_features, tt_ranks, bias=(linear.bias is not None))
    cores = tt_decompose(linear.weight.data, tt_ranks)
    for i, core in enumerate(cores):
        tt.cores[i].data = core
    if linear.bias is not None:
        tt.bias.data = linear.bias.data.clone()
    return tt
