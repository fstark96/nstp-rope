import torch, time, sys
sys.path.insert(0, '/tmp/nstp-v2')

def acc_old(h, positions, D):
    batch, seq_len, _ = h.shape
    bound = torch.zeros_like(h)
    unique_pos = positions.unique()
    for pos in unique_pos:
        mask = (positions == pos)
        if mask.any():
            shifted = h[mask]
            p = pos.item() % D
            if p == 0:
                bound[mask] = shifted
            else:
                bound[mask] = torch.cat([shifted[..., D-p:], shifted[..., :D-p]], dim=-1)
    return bound

def acc_new(h, positions, D):
    batch, seq_len, _ = h.shape
    k_idx = torch.arange(D, device=h.device).view(1, 1, D)
    shifts = positions.view(batch, seq_len, 1)
    idx = (k_idx - shifts) % D  # RIGHT cyclic shift: (k - pos) % D
    return torch.gather(h, 2, idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
B, N, D = 2, 256, 4096
h = torch.randn(B, N, D, device=device)
pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

acc_old(h, pos, D); acc_new(h, pos, D); torch.cuda.synchronize()

t0 = time.time()
for _ in range(20): acc_old(h, pos, D); torch.cuda.synchronize()
t_old = (time.time()-t0)/20*1000

t0 = time.time()
for _ in range(20): acc_new(h, pos, D); torch.cuda.synchronize()
t_new = (time.time()-t0)/20*1000

diff = (acc_old(h, pos, D) - acc_new(h, pos, D)).abs().max().item()
print(f"Old (Python loop): {t_old:.2f}ms")
print(f"New (torch.gather): {t_new:.2f}ms")
print(f"Speedup: {t_old/t_new:.1f}x")
print(f"Correctness: {diff:.2e}")
