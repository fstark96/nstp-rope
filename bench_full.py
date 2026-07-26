import torch, time, sys, math
sys.path.insert(0, '/tmp/nstp-v2')
from nstp_core.model import NSTPModel, NSTPConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
B, N = 2, 256

# NSTP config matching the v2 bundle
config = NSTPConfig(
    vocab_size=1000, d_model=128, num_layers=2, num_heads=4,
    hsa_dim=1024, hsa_binary=True, num_experts=2, top_k=2, d_ff=256,
    router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1])

model = NSTPModel(config).to(device)
x = torch.randint(0, 1000, (B, N), device=device)

# Warmup
model(x)
torch.cuda.synchronize()

# Benchmark
t0 = time.time()
for _ in range(20):
    model(x)
    torch.cuda.synchronize()
t_nstp = (time.time() - t0) / 20 * 1000

# Standard transformer
class StdBlock(torch.nn.Module):
    def __init__(self, d=128, heads=4, dff=256):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3*d)
        self.proj = torch.nn.Linear(d, d)
        self.norm2 = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, dff)
        self.fc2 = torch.nn.Linear(dff, d)
    def forward(self, x):
        h = self.norm1(x)
        q,k,v = self.qkv(h).chunk(3, dim=-1)
        q,k,v = [t.view(x.shape[0],x.shape[1],-1) for t in (q,k,v)]
        a = (q@k.transpose(-2,-1))/math.sqrt(128)
        x = x + self.proj((a.softmax(-1)@v).reshape(x.shape[0],x.shape[1],-1))
        h = self.norm2(x)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))

std = torch.nn.Sequential(StdBlock(), StdBlock()).to(device)
x_emb = torch.randn(B, N, 128, device=device)
std(x_emb); torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    std(x_emb)
    torch.cuda.synchronize()
t_std = (time.time() - t0) / 20 * 1000

print(f"NSTP:     {t_nstp:.2f}ms  ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Standard: {t_std:.2f}ms  ({sum(p.numel() for p in std.parameters()):,} params)")
print(f"Ratio:    {t_nstp/t_std:.1f}x")
