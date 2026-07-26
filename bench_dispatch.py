import torch, time, sys
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
E = moe.num_experts
K = moe.top_k
D = moe.d_model

h = torch.randn(B, N, D, device=device)
x_normed = moe.norm(h)
x_flat = x_normed.reshape(-1, D)
gates, indices, _ = moe.router(x_flat)

def dispatch_original(x, gates, indices, experts, E, K, D):
    output = torch.zeros_like(x)
    for e in range(E):
        for k in range(K):
            mask = (indices[:, k] == e)
            if mask.any():
                output[mask] += gates[mask, k].unsqueeze(1) * experts[e](x[mask])
    return output

def dispatch_fused(x, gates, indices, experts, E, K, D):
    # all_out: (E, T, D)
    all_out = torch.stack([experts[e](x) for e in range(E)])
    # gm: (T, E) — gate value per expert per token
    gm = torch.zeros(x.shape[0], E, device=x.device)
    for k in range(K):
        gm.scatter_add_(1, indices[:, k:k+1], gates[:, k:k+1])
    # einsum: 'etd,te->td' means output[t,d] = sum_e all_out[e,t,d] * gm[t,e]
    # But all_out is (E,T,D) not (E,T,D) indexed as etd... 
    # Let me just use matmul instead
    # gm: (T, E), all_out: (E, T, D) -> permute to (E, D, T)
    # No that's wrong. Let me think.
    # output[t,d] = sum_e gm[t,e] * all_out[e,t,d]
    # Reshape all_out to (E, T*D), then gm @ all_out -> (T, D)
    T = x.shape[0]
    all_flat = all_out.reshape(E, T * D)  # (E, T*D)
    out_flat = gm @ all_flat  # (T, E) @ (E, T*D) -> (T, T*D)?? NO!
    # That's wrong too. Let me just do it properly.
    # all_out[e, t, d] -> for each t, we want sum_e gm[t,e] * all_out[e,t,d]
    # This is: output = torch.bmm(gm.unsqueeze(1), all_out.permute(1,0,2)).squeeze(1)
    # gm: (T, 1, E), all_out.permute: (T, E, D) -> bmm -> (T, 1, D) -> squeeze -> (T, D)
    output = torch.bmm(gm.unsqueeze(1), all_out.permute(1, 0, 2)).squeeze(1)
    return output

def profile_fn(fn, n=30):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn(); torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

dispatch_original(x_flat, gates, indices, moe.experts, E, K, D)
dispatch_fused(x_flat, gates, indices, moe.experts, E, K, D)
torch.cuda.synchronize()

t_orig = profile_fn(lambda: dispatch_original(x_flat, gates, indices, moe.experts, E, K, D))
t_fused = profile_fn(lambda: dispatch_fused(x_flat, gates, indices, moe.experts, E, K, D))

out_orig = dispatch_original(x_flat, gates, indices, moe.experts, E, K, D)
out_fused = dispatch_fused(x_flat, gates, indices, moe.experts, E, K, D)
diff = (out_orig - out_fused).abs().max().item()

print(f"Original (loop):  {t_orig:.2f}ms")
print(f"Fused (scatter):  {t_fused:.2f}ms")
print(f"Speedup:          {t_orig/t_fused:.1f}x")
print(f"Correctness:      {diff:.2e}")
