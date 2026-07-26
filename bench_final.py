import torch, time, sys
sys.path.insert(0, '/tmp/nstp-v2')
from nstp_core.model import NSTPModel, NSTPConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = NSTPConfig(
    vocab_size=1000, d_model=128, num_layers=2, num_heads=4,
    hsa_dim=1024, hsa_binary=True, num_experts=2, top_k=2, d_ff=256,
    router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1])
model = NSTPModel(config).to(device)
params = sum(p.numel() for p in model.parameters())

B, N = 4, 256
x = torch.randint(0, 1000, (B, N), device=device)

def profile(n=100):
    for _ in range(10): model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): model(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

ms = profile()
print(f"Model: {params:,} params")
print(f"Full forward: {ms:.2f}ms")
print(f"  vs Standard Transformer: ~0.7ms")
print(f"  Gap: {ms/0.7:.1f}x slower")
