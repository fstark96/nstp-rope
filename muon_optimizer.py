"""
muon_optimizer.py — Muon optimizer for NSTP-Ω V3
Adapted from Karpathy's nanochat/autoresearch.

Muon = Momentum Orthogonalized by Newton-Schulz iteration.
Used for 2D matrix params (transformer weights, FFN, attention).
Embeddings, scalars, biases → AdamW (kept separate).

Polar Express coefficients approximate orthogonalization via:
  X ← a*X + X @ (b*X^T@X + c*(X^T@X)^2)

NorMuon adds per-parameter LR scaling via second momentum buffer,
ensuring all matrix params get updates of similar magnitude.

Cautious weight decay: only apply WD when gradient agrees with weight sign.
This prevents decay from hurting "growing" weights.
"""
import torch
from torch import Tensor
import math
from typing import List


# Polar Express coefficients (5-step Newton-Schulz iteration)
# These approximate matrix orthogonalization X ≈ UV^T from X = USV^T
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Orthogonalize G via 5-step Newton-Schulz iteration.
    G: (..., m, n) where m,n are matrix dims.
    Returns G with rows (or cols) approximately orthogonal.
    """
    assert G.ndim >= 2
    X = G.bfloat16()

    # Transpose if needed so we always orthogonalize the larger dim
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize so spectral norm ≤ 1 (the iteration only converges there)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    # 5 polar express iterations
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # Transpose back
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D matrix parameters.

    Key features:
    - Momentum-based gradient buffering
    - Newton-Schulz orthogonalization of gradient
    - NorMuon-style variance reduction (per-parameter LR scaling)
    - Cautious weight decay (only when sign(grad) == sign(param))

    Reference: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
        cautious_wd: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            beta2=beta2,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            beta2 = group["beta2"]
            wd = group["weight_decay"]
            cautious = group["cautious_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                # Only handle 2D+ tensors (matrices)
                if g.ndim < 2:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Nesterov momentum
                buf.lerp_(g, 1 - momentum)
                g_eff = g.lerp_(buf, momentum) if nesterov else buf

                # Orthogonalize via Newton-Schulz
                g_orth = zeropower_via_newtonschulz5(g_eff, steps=ns_steps)

                # NorMuon variance reduction: scale per-row/col so updates have similar magnitude
                # Compute row-wise (or col-wise) squared mean
                if g_orth.size(-2) >= g_orth.size(-1):
                    # Tall matrix: scale rows
                    v_mean = g_orth.float().pow(2).mean(dim=-1, keepdim=True)
                else:
                    # Wide matrix: scale cols
                    v_mean = g_orth.float().pow(2).mean(dim=-2, keepdim=True)

                if "second_momentum_buffer" not in state:
                    state["second_momentum_buffer"] = torch.zeros_like(v_mean)

                v_buf = state["second_momentum_buffer"]
                v_buf.lerp_(v_mean, 1 - beta2)

                # Scale: 1/sqrt(v_buf) gives per-row/col LR
                # Then rescale so total update magnitude matches AdamW scale (~sqrt(num_params))
                scale = v_buf.clamp_min(1e-10).rsqrt()

                # Compute rescale factor
                target_norm = math.sqrt(max(g_orth.numel(), 1))
                current_norm = (g_orth.float().pow(2).sum().sqrt() + 1e-10)
                rescale = target_norm / current_norm

                # Final gradient: orthogonalized * per-row/col scale * rescale
                update = g_orth * (scale * rescale).to(g_orth.dtype)

                # Cautious weight decay: only decay where sign(grad) == sign(param)
                if wd > 0:
                    if cautious:
                        mask = (update * p.data) >= 0
                        # wd applies to params where update and param agree (i.e. growing)
                        p.data.add_(update, alpha=-lr)
                        p.data.add_(p.data * mask, alpha=-lr * wd)
                    else:
                        p.data.mul_(1 - lr * wd)
                        p.data.add_(update, alpha=-lr)
                else:
                    p.data.add_(update, alpha=-lr)


def split_params_for_muon(model: torch.nn.Module, lr_scale: dict = None):
    """
    Split model parameters into Muon (2D matrices) and AdamW (everything else).

    Returns:
        muon_params: list of (name, param) — 2D weight matrices (excluding embeddings, lm_head)
        adamw_params: list of (name, param) — 1D params, embeddings, lm_head, biases, norms

    The Muon paper recommends lr_scale ∝ 1/sqrt(d_model).
    """
    if lr_scale is None:
        lr_scale = {}

    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Skip if explicitly assigned
        if any(kw in name for kw in lr_scale.get("_skip", [])):
            adamw_params.append((name, p))
            continue

        # Muon: 2D matrices, not embeddings/lm_head
        is_matrix = p.ndim >= 2
        is_embed = "embed" in name or "wte" in name or "lm_head" in name or "head" in name
        is_norm = "norm" in name or "LayerNorm" in name
        is_scalar = p.ndim < 2

        if is_matrix and not is_embed and not is_norm:
            muon_params.append((name, p))
        else:
            adamw_params.append((name, p))

    return muon_params, adamw_params


if __name__ == "__main__":
    # Quick test
    print("Testing Muon optimizer...")

    # Simple test: minimize ||Wx - y||^2
    torch.manual_seed(0)
    W = torch.randn(64, 32, requires_grad=True)
    x = torch.randn(32, 16)
    y = torch.randn(64, 16)

    opt = Muon([W], lr=0.02, momentum=0.95, weight_decay=0.01)
    losses = []
    for step in range(100):
        opt.zero_grad()
        loss = ((W @ x - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Reduction:    {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    assert losses[-1] < losses[0] / 10, "Muon failed to converge"
    print("  ✓ Muon converges")

    # Test split
    from nstp_omega import NSTPOmega, NSTPOmegaConfig
    config = NSTPOmegaConfig(vocab_size=1000, d_model=128, num_layers=2, num_heads=2)
    model = NSTPOmega(config)

    muon_p, adamw_p = split_params_for_muon(model)
    total = sum(p.numel() for _, p in muon_p) + sum(p.numel() for _, p in adamw_p)
    model_total = sum(p.numel() for p in model.parameters())

    print(f"\n  Model total: {model_total:,} params")
    print(f"  Muon:    {len(muon_p)} tensors, {sum(p.numel() for _, p in muon_p):,} params")
    print(f"  AdamW:   {len(adamw_p)} tensors, {sum(p.numel() for _, p in adamw_p):,} params")
    print(f"  Coverage: {(sum(p.numel() for _, p in muon_p) + sum(p.numel() for _, p in adamw_p)) / model_total * 100:.1f}%")

    print("\n✅ Muon optimizer working!")
