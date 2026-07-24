"""
Verify the pos_embedding fix: print params + time forward.
"""
import sys
sys.path.insert(0, 'C:/Users/online/NSTP')

import torch
import time
from nstp_core import NSTPModel, NSTPConfig
from benchmarks.benchmark import StandardTransformer

torch.manual_seed(0)
device = 'cpu'  # CPU for portability; you can switch to cuda if available

print("=" * 60)
print("NSTP — post pos_embedding removal")
print("=" * 60)

# Same architecture you reportedly benchmarked
nstp_config = NSTPConfig(
    vocab_size=2000,
    d_model=128,
    num_layers=2,
    num_heads=4,
    hsa_dim=1024,
    hsa_bind_mode='xor',
    hsa_binary=True,
    num_experts=2,
    top_k=2,
    d_ff=512,
    router_tt_ranks=[1, 16, 16, 1],
    expert_tt_ranks=[1, 8, 8, 8, 1],
    embedding_tt_ranks=[1, 8, 8, 1],
    use_tt_embedding=True,
)
nstp = NSTPModel(nstp_config).to(device)

# Param breakdown
print("\nParam breakdown:")
total = 0
for name, p in nstp.named_parameters():
    print(f"  {name:50} {str(p.shape):25} {p.numel():>12,}")
    total += p.numel()
print(f"  {'TOTAL':50} {'-'*25:>25} {total:>12,}  <-- {total/1e6:.2f}M")

# Standard transformer for comparison
print("\nStandard Transformer (param breakdown):")
std_layers = nn = torch.nn
std = torch.nn.Sequential(*[
    StandardTransformer(
        d_model=nstp_config.d_model,
        num_heads=nstp_config.num_heads,
        d_ff=nstp_config.d_ff,
        dropout=0.0,
    )
    for _ in range(nstp_config.num_layers)
]).to(device)
total_std = sum(p.numel() for p in std.parameters())
print(f"  Standard total: {total_std:,}  ({total_std/1e3:.1f}K)")

print(f"\nParam ratio: NSTP / Std = {total/total_std:.1f}×")

# Forward benchmark
B, N = 2, 256
x_nstp = torch.randint(0, nstp_config.vocab_size, (B, N), device=device)
x_std = torch.randn(B, N, nstp_config.d_model, device=device)

# Warmup
for _ in range(3):
    with torch.no_grad():
        nstp.eval()
        _ = nstp(x_nstp)
        std.eval()
        _ = std(x_std)

# Time NSTP
nstp.eval()
ts = []
with torch.no_grad():
    for _ in range(20):
        t0 = time.perf_counter()
        out = nstp(x_nstp)
        ts.append(time.perf_counter() - t0)
nstp_time = sum(ts) / len(ts) * 1000
print(f"\nNSTP forward: {nstp_time:.2f}ms (avg of 20)")

# Time Standard
ts = []
with torch.no_grad():
    for _ in range(20):
        t0 = time.perf_counter()
        out = std(x_std)
        ts.append(time.perf_counter() - t0)
std_time = sum(ts) / len(ts) * 1000
print(f"Standard forward: {std_time:.2f}ms (avg of 20)")

print(f"\nSpeed ratio: NSTP / Std = {nstp_time/std_time:.1f}× slower")
print(f"Param ratio: NSTP / Std = {total/total_std:.1f}× bigger")
