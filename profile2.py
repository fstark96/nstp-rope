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
x = torch.randint(0, 1000, (B, N), device=device)

def profile_fn(fn, n=20):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn(); torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

h = model.embedding(x)
positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

# Block 0
t_blk0_attn = profile_fn(lambda: model.blocks[0].attention(h))
out0 = model.blocks[0].attention(h)
h2 = out0[0] if isinstance(out0, tuple) else out0
t_blk0_moe = profile_fn(lambda: model.blocks[0].moe(h2))
out_moe0 = model.blocks[0].moe(h2)
h3 = out_moe0[0] if isinstance(out_moe0, tuple) else out_moe0

# Block 1
t_blk1_attn = profile_fn(lambda: model.blocks[1].attention(h3))
out1 = model.blocks[1].attention(h3)
h4 = out1[0] if isinstance(out1, tuple) else out1
t_blk1_moe = profile_fn(lambda: model.blocks[1].moe(h4))
out_moe1 = model.blocks[1].moe(h4)
h5 = out_moe1[0] if isinstance(out_moe1, tuple) else out_moe1

t_norm = profile_fn(lambda: model.norm(h5))
h6 = model.norm(h5)
t_head = profile_fn(lambda: model.lm_head(h6))

t_full = profile_fn(lambda: model(x))
total_comp = t_blk0_attn + t_blk0_moe + t_blk1_attn + t_blk1_moe + t_norm + t_head

print(f"Block0 HSA:     {t_blk0_attn:.2f}ms")
print(f"Block0 MoE:     {t_blk0_moe:.2f}ms")
print(f"Block1 HSA:     {t_blk1_attn:.2f}ms")
print(f"Block1 MoE:     {t_blk1_moe:.2f}ms")
print(f"Norm+Head:      {t_norm + t_head:.2f}ms")
print(f"Total comps:    {total_comp:.2f}ms")
print(f"Full model:     {t_full:.2f}ms")
print(f"Overhead:       {t_full - total_comp:.2f}ms")
print(f"Standard:       ~0.7ms")
print(f"Ratio:          {t_full/0.7:.1f}x")
