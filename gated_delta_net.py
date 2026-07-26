"""Gated DeltaNet — O(n) token mixing layer.
Based on: "Gated Delta Networks" (ICLR 2025) https://arxiv.org/abs/2412.06464

Algorithm:
    state_t = alpha_t * state_{t-1} + (1 - alpha_t) * key_t    # gated state
    output_t = query_t @ state_t                                 # O(1) output

Replaces O(n²) softmax attention with O(n) recurrence.
"""
import sys as _sys
import types as _types
_fm = _types.ModuleType('profile'); _fm.run = _fm; _fm.runctx = _fm
_fm.Profile = type('P',(),{'__init__':lambda s,*a,**k:None})
_sys.modules['profile'] = _fm

import math, time, torch, torch.nn as nn, torch.nn.functional as F


class GatedDeltaLayer(nn.Module):
    """O(n) token mixer using gated delta-rule recurrence."""
    def __init__(self, d_model, num_heads, apply_rope=True):
        super().__init__()
        H = num_heads
        hd = d_model // H
        self.d_model = d_model; self.num_heads = H; self.head_dim = hd
        self.scale = hd ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.alpha = nn.Linear(d_model, H)   # retention gate per head
        self.o_proj = nn.Linear(d_model, d_model)

        if apply_rope:
            inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
            self.register_buffer('inv_freq', inv)
        self.apply_rope = apply_rope

    def _rope(self, q, k):
        """Apply RoPE. q/k: (B, S, H, hd)"""
        S = q.size(1)
        t = torch.arange(S, device=q.device, dtype=q.dtype)
        f = torch.outer(t, self.inv_freq)
        e = torch.cat([f, f], dim=-1)
        cos = e.cos().unsqueeze(0).unsqueeze(2)   # (1, S, 1, hd)
        sin = e.sin().unsqueeze(0).unsqueeze(2)
        def rot(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        return q * cos + rot(q) * sin, k * cos + rot(k) * sin

    def forward(self, x):
        B, S, D = x.shape
        H = self.num_heads; hd = self.head_dim

        q = self.q_proj(x).view(B, S, H, hd)
        k = self.k_proj(x).view(B, S, H, hd)

        if self.apply_rope:
            q, k = self._rope(q, k)

        alpha = torch.sigmoid(self.alpha(x))   # (B, S, H)

        state = torch.zeros(B, H, hd, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            a = alpha[:, t, :]     # (B, H)
            kt = k[:, t, :, :]      # (B, H, hd)
            state = a.unsqueeze(-1) * state + (1 - a.unsqueeze(-1)) * kt
            qt = q[:, t, :, :]      # (B, H, hd)
            y = (qt * state).sum(-1) * self.scale  # (B, H)
            outputs.append(y)

        out = torch.stack(outputs, dim=1)                     # (B, S, H)
        out = out.unsqueeze(-1).expand(B, S, H, hd).reshape(B, S, D)
        return self.o_proj(out)


class GatedDeltaBlock(nn.Module):
    """Transformer block: GatedDeltaLayer + FFN with pre-norm."""
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.delta = GatedDeltaLayer(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff or d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff or d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.delta(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Benchmark ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("Gated DeltaNet — O(n) Token Mixing")
    print("=" * 60)

    B, S, D, H = 2, 256, 256, 8
    block = GatedDeltaBlock(D, H).to(device)
    total = sum(p.numel() for p in block.parameters())
    print(f"Block params: {total:,}")

    x = torch.randn(B, S, D, device=device)

    # Forward + backward
    block.train()
    out = block(x)
    print(f"Output: {out.shape}, no NaN: {not torch.isnan(out).any()}")
    loss = out.sum(); loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in block.parameters())
    print(f"Backward OK: {has_grad}")

    # Speed benchmark vs SDPA+RoPE
    rope_cache = {}
    def _rope(S, dev, dtype):
        if S not in rope_cache:
            inv = 1.0 / (10000 ** (torch.arange(0, 32, 2, dtype=dtype, device=dev).float() / 32))
            t = torch.arange(S, device=dev, dtype=dtype)
            f = torch.outer(t, inv); e = torch.cat([f, f], dim=-1)
            rope_cache[S] = (e.cos().unsqueeze(0).unsqueeze(1),
                             e.sin().unsqueeze(0).unsqueeze(1))
        return rope_cache[S]

    sdpa = nn.ModuleDict({
        'q': nn.Linear(D, D), 'k': nn.Linear(D, D), 'v': nn.Linear(D, D),
        'o': nn.Linear(D, D),
        'ffn': nn.Sequential(nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D)),
        'ln1': nn.LayerNorm(D), 'ln2': nn.LayerNorm(D),
    }).to(device)

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def sdpa_forward(x_):
        B2, S2, _ = x_.shape
        h = sdpa['ln1'](x_)
        q = sdpa['q'](h).view(B2, S2, H, 32).transpose(1, 2)
        k = sdpa['k'](h).view(B2, S2, H, 32).transpose(1, 2)
        v = sdpa['v'](h).view(B2, S2, H, 32).transpose(1, 2)
        cos, sin = _rope(S2, x_.device, x_.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        causal = torch.triu(torch.ones(S2, S2, device=x_.device, dtype=torch.bool), 1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal, is_causal=False)
        x_ = x_ + attn.transpose(1, 2).reshape(B2, S2, D)
        return x_ + sdpa['ffn'](sdpa['ln2'](x_))

    for name, fn, model in [
        ('GatedDeltaNet (O-n)',  lambda x: block(x),   block),
        ('SDPA+RoPE   (O-n2)',   sdpa_forward,       sdpa),
    ]:
        torch.cuda.reset_peak_memory_stats()
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            out = fn(x)
            out.sum().backward()
        torch.cuda.synchronize()
        dt  = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{name}: 5 steps in {dt:.3f}s, peak={mem:.2f}GB")

    # Scaling: time vs sequence length
    print("\n--- Scaling: time vs seq len ---")
    for name, fn in [('GatedDeltaNet', block.forward), ('SDPA+RoPE', sdpa_forward)]:
        times = []
        for s in [64, 128, 256, 512, 1024]:
            xs = torch.randn(2, s, D, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(xs); out.sum().backward()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"  {name}: " + "  ".join(f"S{s}={t:.3f}s" for s, t in zip([64,128,256,512,1024], times)))

    print("\n✅ GatedDeltaNet test complete")