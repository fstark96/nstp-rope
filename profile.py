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

# Profile each block
def profile_fn(name, fn, n=10):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn(); torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

# Warmup all
for block in model.blocks:
    block.attention(model.embedding(x))
    block.moe(torch.randn(B, N, 128, device=device))

# Profile components
h = model.embedding(x)
positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

t_emb = profile_fn('Embedding', lambda: model.embedding(x))
t_pos = profile_fn('Positions', lambda: positions)

# Profile attention components per head
attn = model.blocks[0].attention
h_enc = attn.encoders[0](h)
t_enc = profile_fn('Encoder', lambda: attn.encoders[0](h))

h_acc = attn.accumulators[0](h_enc, positions)
t_acc = profile_fn('Accumulator', lambda: attn.accumulators[0](h_enc, positions))

t_retr = profile_fn('Retrieval', lambda: attn._retrieve_xor_vectorized(h_acc, positions, B, N, attn.head_dim))

t_den = profile_fn('Denoiser', lambda: attn.denoisers[0](h_acc))

# Full attention
t_attn = profile_fn('Full HSA Attn', lambda: attn(h))

# MoE
t_moe = profile_fn('MoE', lambda: model.blocks[0].moe(h))

# Full model
t_full = profile_fn('Full model', lambda: model(x))

print(f"Component timings (256d, 256ctx, GPU):")
print(f"  Embedding:    {t_emb:.2f}ms")
print(f"  Encoder:      {t_enc:.2f}ms")
print(f"  Accumulator:  {t_acc:.2f}ms")
print(f"  Retrieval:    {t_retr:.2f}ms")
print(f"  Denoiser:     {t_den:.2f}ms")
print(f"  Full HSA:     {t_attn:.2f}ms")
print(f"  MoE:          {t_moe:.2f}ms")
print(f"  Full model:   {t_full:.2f}ms")
