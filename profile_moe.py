import torch, time, sys, math
sys.path.insert(0, '/tmp/nstp-v2')
from nstp_core.model import NSTPModel, NSTPConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
B, N = 2, 256

config = NSTPConfig(
    vocab_size=1000, d_model=128, num_layers=2, num_heads=4,
    hsa_dim=1024, hsa_binary=True, num_experts=2, top_k=2, d_ff=256,
    router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1])

model = NSTPModel(config).to(device)
moe = model.blocks[0].moe

def profile_fn(fn, n=30):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn(); torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

h = torch.randn(B, N, 128, device=device)
x_flat = h.reshape(-1, 128)

# 1. Router forward
t_router = profile_fn(lambda: moe.router(x_flat))

# 2. Single expert (64 tokens)
single_expert = moe.experts[0]
t_single = profile_fn(lambda: single_expert(x_flat[:64]))

# 3. All experts on all tokens (no dispatch)
t_all_experts = profile_fn(lambda: torch.stack([e(x_flat) for e in moe.experts]))

# 4. Full MoE (with dispatch loop)
t_moe = profile_fn(lambda: moe(h))

# 5. Dispatch-only (no expert compute)
def dispatch_only():
    x_normed = moe.norm(h)
    x_flat_l = x_normed.reshape(-1, 128)
    gates, indices, _ = moe.router(x_flat_l)
    output = torch.zeros_like(x_flat_l)
    for expert_idx in range(moe.num_experts):
        for k in range(moe.top_k):
            k_mask = (indices[:, k] == expert_idx)
            if k_mask.any():
                token_gates = gates[k_mask, k].unsqueeze(1)
                expert_input = x_flat_l[k_mask]
                expert_output = moe.experts[expert_idx](expert_input)
                output[k_mask] += token_gates * expert_output
    return output

t_dispatch = profile_fn(dispatch_only)

# Compute cost
compute_per_expert = t_all_experts / 4
compute_total = compute_per_expert * 4

print("=== MoE Profiling ===")
print(f"Router:           {t_router:.2f}ms")
print(f"Single expert:    {t_single:.2f}ms")
print(f"All experts:      {t_all_experts:.2f}ms")
print(f"Dispatch loop:    {t_dispatch:.2f}ms")
print(f"Full MoE:         {t_moe:.2f}ms")
print()
print(f"TTLinear compute: {compute_total:.2f}ms")
print(f"Dispatch overhead: {t_dispatch - compute_total:.2f}ms")
print()
verdict = "DISPATCH" if (t_dispatch - compute_total) > compute_total else "COMPUTE"
print(f"VERDICT: {verdict} bottleneck")
