"""
NSTP Benchmark Suite
Compares NSTP (HSA + TT-CER) against standard Transformer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics


@dataclass
class BenchmarkResult:
    name: str
    forward_time_ms: float
    backward_time_ms: float
    peak_memory_mb: float
    params: int
    flops: int
    compression_ratio: float


class StandardTransformer(nn.Module):
    """Standard Transformer block for comparison."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Multi-head attention
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        residual = x
        x = self.norm1(x)
        
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # [B, N, H, D]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.d_model)
        out = self.proj(out)
        x = residual + out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = self.dropout(x)
        x = residual + x
        
        return x


class StandardTransformerMoE(nn.Module):
    """Standard Transformer with MoE (no TT compression)."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Multi-head attention
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        
        # MoE
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        
        # Attention
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.d_model)
        out = self.proj(out)
        x = residual + out
        
        # MoE
        residual = x
        x = self.norm2(x)
        x_flat = x.reshape(-1, self.d_model)
        
        # Router
        logits = self.router(x_flat)
        gates, indices = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(gates, dim=-1)
        
        # Expert dispatch
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = indices[:, k]
            gate = gates[:, k].unsqueeze(1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](x_flat[mask])
        
        out = out.reshape(B, N, self.d_model)
        out = self.dropout(out)
        x = residual + out
        
        # Aux loss
        expert_fraction = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            expert_fraction += torch.bincount(indices[:, k], minlength=self.num_experts).float()
        expert_fraction = expert_fraction / (B * N * self.top_k)
        aux_loss = self.num_experts * (expert_fraction ** 2).sum()
        
        return x, aux_loss


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate FLOPs for a forward pass."""
    # Rough estimation
    return 0


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = 'cuda',
) -> BenchmarkResult:
    """
    Benchmark a model's forward and backward pass.
    
    Args:
        model: Model to benchmark
        input_shape: (batch, seq_len, d_model)
        num_warmup: Warmup iterations
        num_runs: Benchmark iterations
        device: Device to run on
    
    Returns:
        BenchmarkResult with timing and memory stats
    """
    model = model.to(device)
    model.train()
    
    # Create input
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    
    # Warmup
    for _ in range(num_warmup):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        model.zero_grad()
    
    # Benchmark forward
    torch.cuda.synchronize() if device == 'cuda' else None
    forward_times = []
    
    for _ in range(num_runs):
        model.zero_grad()
        x.grad = None
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)
    
    # Benchmark backward
    backward_times = []
    
    for _ in range(num_runs):
        model.zero_grad()
        x.grad = None
        
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        loss.backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)
    
    # Memory
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory = 0
    
    # Stats
    mean_forward = statistics.mean(forward_times) * 1000  # ms
    mean_backward = statistics.mean(backward_times) * 1000  # ms
    params = count_params(model)
    
    return BenchmarkResult(
        name=model.__class__.__name__,
        forward_time_ms=mean_forward,
        backward_time_ms=mean_backward,
        peak_memory_mb=peak_memory,
        params=params,
        flops=0,  # TODO: compute
        compression_ratio=1.0,
    )


def benchmark_standard_vs_nstp(
    d_model: int = 512,
    seq_len: int = 512,
    num_layers: int = 4,
    num_experts: int = 8,
    batch_size: int = 2,
    device: str = 'cuda',
) -> Dict[str, BenchmarkResult]:
    """Compare standard transformer vs NSTP."""
    
    results = {}
    
    # Standard Transformer
    print(f"Benchmarking Standard Transformer ({d_model}d, {seq_len}ctx, {num_layers}L)...")
    std_model = nn.Sequential(*[
        StandardTransformer(
            d_model=d_model,
            num_heads=8,
            d_ff=2048,
            dropout=0.1,
        )
        for _ in range(num_layers)
    ])
    results['standard'] = benchmark_model(
        std_model, (batch_size, seq_len, d_model), device=device
    )
    print(f"  Forward: {results['standard'].forward_time_ms:.2f}ms")
    print(f"  Backward: {results['standard'].backward_time_ms:.2f}ms")
    print(f"  Memory: {results['standard'].peak_memory_mb:.1f}MB")
    print(f"  Params: {results['standard'].params:,}")
    
    # Standard MoE Transformer
    print(f"\nBenchmarking Standard MoE Transformer ({num_experts} experts)...")
    std_moe = nn.Sequential(*[
        StandardTransformerMoE(
            d_model=d_model,
            num_heads=8,
            d_ff=2048,
            num_experts=num_experts,
            top_k=2,
            dropout=0.1,
        )
        for _ in range(num_layers)
    ])
    results['standard_moe'] = benchmark_model(
        std_moe, (batch_size, seq_len, d_model), device=device
    )
    print(f"  Forward: {results['standard_moe'].forward_time_ms:.2f}ms")
    print(f"  Backward: {results['standard_moe'].backward_time_ms:.2f}ms")
    print(f"  Memory: {results['standard_moe'].peak_memory_mb:.1f}MB")
    print(f"  Params: {results['standard_moe'].params:,}")
    
    # NSTP Model
    print(f"\nBenchmarking NSTP Model...")
    from nstp_core import NSTPModel, NSTPConfig
    
    nstp_config = NSTPConfig(
        vocab_size=1000,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=8,
        hsa_dim=16384,
        hsa_bind_mode='xor',
        hsa_binary=True,
        num_experts=num_experts,
        top_k=2,
        d_ff=2048,
        router_tt_ranks=[1, 16, 16, 1],
        expert_tt_ranks=[1, 16, 16, 16, 1],
        capacity_factor=1.25,
        dropout=0.1,
    )
    nstp_model = NSTPModel(nstp_config)
    results['nstp'] = benchmark_model(
        nstp_model, (batch_size, seq_len, d_model), device=device
    )
    print(f"  Forward: {results['nstp'].forward_time_ms:.2f}ms")
    print(f"  Backward: {results['nstp'].backward_time_ms:.2f}ms")
    print(f"  Memory: {results['nstp'].peak_memory_mb:.1f}MB")
    print(f"  Params: {results['nstp'].params:,}")
    
    # Compute speedups
    print("\n=== SPEEDUP COMPARISON ===")
    std_fwd = results['standard'].forward_time_ms
    nstp_fwd = results['nstp'].forward_time_ms
    print(f"NSTP vs Standard: {std_fwd / nstp_fwd:.2f}x faster forward")
    
    std_mem = results['standard'].peak_memory_mb
    nstp_mem = results['nstp'].peak_memory_mb
    if nstp_mem > 0:
        print(f"NSTP vs Standard: {std_mem / nstp_mem:.2f}x less memory")
    
    std_params = results['standard'].params
    nstp_params = results['nstp'].params
    print(f"NSTP vs Standard: {std_params / nstp_params:.2f}x fewer params")
    
    return results


def benchmark_scaling(
    seq_lengths: List[int] = [256, 512, 1024, 2048, 4096, 8192],
    d_model: int = 512,
    num_layers: int = 2,
    batch_size: int = 1,
    device: str = 'cuda',
) -> Dict[int, Dict[str, BenchmarkResult]]:
    """Benchmark how models scale with sequence length."""
    
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n=== Sequence Length: {seq_len} ===")
        try:
            results[seq_len] = benchmark_standard_vs_nstp(
                d_model=d_model,
                seq_len=seq_len,
                num_layers=num_layers,
                batch_size=batch_size,
                device=device,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at seq_len={seq_len}")
                break
            raise
    
    return results


def print_scaling_results(results: Dict[int, Dict[str, BenchmarkResult]]):
    """Print scaling comparison table."""
    print("\n=== SCALING ANALYSIS ===")
    print(f"{'Seq Len':>8} | {'Std Fwd (ms)':>12} | {'NSTP Fwd (ms)':>12} | {'Speedup':>8} | {'Std Mem (MB)':>12} | {'NSTP Mem (MB)':>12}")
    print("-" * 90)
    
    for seq_len, res in results.items():
        std_fwd = res['standard'].forward_time_ms
        nstp_fwd = res['nstp'].forward_time_ms
        speedup = std_fwd / nstp_fwd if nstp_fwd > 0 else 0
        std_mem = res['standard'].peak_memory_mb
        nstp_mem = res['nstp'].peak_memory_mb
        
        print(f"{seq_len:>8} | {std_fwd:>12.1f} | {nstp_fwd:>12.1f} | {speedup:>8.2f}x | {std_mem:>12.1f} | {nstp_mem:>12.1f}")


def theoretical_flops_comparison(
    d_model: int = 512,
    seq_len: int = 1024,
    num_layers: int = 12,
    num_experts: int = 8,
    top_k: int = 2,
    hsa_dim: int = 16384,
    tt_rank: int = 16,
) -> Dict[str, float]:
    """
    Theoretical FLOPs comparison between standard and NSTP.
    """
    B = 1  # per sample
    
    # Standard Attention: O(B * num_layers * num_heads * seq_len^2 * head_dim)
    # MHA FLOPs ≈ 4 * B * L * H * N^2 * d_h
    std_attn_flops = 4 * B * num_layers * (d_model // 64) * seq_len ** 2 * 64
    
    # Standard MoE FFN: O(B * L * N * k * d_model * d_ff)
    # Each token goes to k experts, each expert does 2 * d_model * d_ff
    d_ff = 4 * d_model
    std_moe_flops = 2 * B * num_layers * seq_len * top_k * d_model * d_ff
    
    std_total = std_attn_flops + std_moe_flops
    
    # NSTP HSA: O(B * L * N * HSA_dim)
    nstp_hsa_flops = B * num_layers * seq_len * hsa_dim
    
    # NSTP TT-CER: O(B * L * N * k * d_model * tt_rank^2)
    nstp_moe_flops = 2 * B * num_layers * seq_len * top_k * d_model * (tt_rank ** 2)
    
    nstp_total = nstp_hsa_flops + nstp_moe_flops
    
    return {
        'std_attention_flops': std_attn_flops,
        'std_moe_flops': std_moe_flops,
        'std_total_flops': std_total,
        'nstp_hsa_flops': nstp_hsa_flops,
        'nstp_moe_flops': nstp_moe_flops,
        'nstp_total_flops': nstp_total,
        'flops_reduction': std_total / nstp_total,
        'attn_flops_reduction': std_attn_flops / nstp_hsa_flops,
        'moe_flops_reduction': std_moe_flops / nstp_moe_flops,
    }


def print_theoretical_comparison():
    """Print theoretical FLOPs comparison."""
    print("\n=== THEORETICAL FLOPs COMPARISON ===")
    
    configs = [
        {"d_model": 512, "seq_len": 1024, "label": "Small (512d, 1K ctx)"},
        {"d_model": 1024, "seq_len": 2048, "label": "Medium (1Kd, 2K ctx)"},
        {"d_model": 2048, "seq_len": 4096, "label": "Large (2Kd, 4K ctx)"},
        {"d_model": 4096, "seq_len": 8192, "label": "XL (4Kd, 8K ctx)"},
    ]
    
    for cfg in configs:
        flops = theoretical_flops_comparison(
            d_model=cfg["d_model"],
            seq_len=cfg["seq_len"],
            num_layers=12,
            num_experts=8,
        )
        print(f"\n{cfg['label']}:")
        print(f"  Standard Total: {flops['std_total_flops'] / 1e12:.2f} TFLOPs")
        print(f"  NSTP Total:     {flops['nstp_total_flops'] / 1e12:.2f} TFLOPs")
        print(f"  Reduction:      {flops['flops_reduction']:.1f}x")
        print(f"  Attention:      {flops['attn_flops_reduction']:.1f}x")
        print(f"  MoE:            {flops['moe_flops_reduction']:.1f}x")


if __name__ == "__main__":
    # Print theoretical comparison
    print_theoretical_comparison()
    
    # Run benchmarks if CUDA available
    if torch.cuda.is_available():
        print("\n\nRunning GPU benchmarks...")
        results = benchmark_standard_vs_nstp(
            d_model=512,
            seq_len=1024,
            num_layers=4,
            num_experts=8,
            batch_size=2,
            device='cuda',
        )
    else:
        print("\nCUDA not available, skipping GPU benchmarks.")
        print("Running CPU benchmarks (slower)...")
        results = benchmark_standard_vs_nstp(
            d_model=256,
            seq_len=512,
            num_layers=2,
            num_experts=4,
            batch_size=1,
            device='cpu',
        )