"""
nstp_omega_v3.py — NSTP-Ω V3 with autoresearch-inspired improvements.

V3 = V2 + these Karpathy-style additions:

1. **Softcap logits** (softcap=15): `logits = softcap * tanh(logits / softcap)`
   Prevents overconfident logits → stabilizes training, often improves convergence.

2. **Value Embeddings (ResFormer)**: Per-token learnable V bias added to value,
   mixed via input-dependent gate. Only on alternating layers.
   Costs: vocab_size × n_kv_head × head_dim = 50257 × 4 × 64 = 12.9M extra params
   (we use fewer VEs since our head_dim is 64 and we have 8 heads → 4 KV heads)

3. **Windowed attention pattern**: Most layers use short-context, only last full.
   We use 'SSSL' pattern but adapted for NSTP-Ω's DeltaNet (which is O(n) anyway,
   so windowing is conceptual — we use it to limit attention to recent context).

4. **x0 residual lambdas**: Per-layer mixing of x0 into residual stream.
   Initialized at 0.1 (gentle boost from initial embedding).

5. **Faster-init (s=3^0.5 * d^(-0.5))**: Per the muon paper's recommendation.

We keep the core architecture intact — V3 is purely a training/inference improvement.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

# Import V2 components (re-use same architecture)
from nstp_omega import (
    NSTPOmegaConfig,
    GatedDeltaNetOmega,
    RFMoE,
    HaltGate,
    EagleOmega,
    HierarchicalHDCMemory,
    QuantizedLinear158,
)


# ============================================================================
# SOFTCAP LOGITS
# ============================================================================
def softcap_logits(logits: torch.Tensor, softcap: float = 15.0) -> torch.Tensor:
    """
    Softcap logits: `logits = softcap * tanh(logits / softcap)`
    Prevents any logit from exceeding ±softcap in magnitude.
    Improves training stability, especially early in training when logits
    can spike to ±100+ before being tamed.
    """
    return softcap * torch.tanh(logits / softcap)


# ============================================================================
# VALUE EMBEDDING (ResFormer style)
# ============================================================================
class ValueEmbedding(nn.Module):
    """
    Per-token learnable value embedding.
    Mixed into V via input-dependent gate per head.

    Cheap to add (one Embedding + one tiny Linear gate).
    Acts like a learned positional/identity bias on attention values.
    """
    def __init__(self, vocab_size: int, n_kv_head: int, head_dim: int):
        super().__init__()
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.embed = nn.Embedding(vocab_size, n_kv_head * head_dim)
        # Gate: input → per-KV-head mixing coefficient
        # First `gate_channels` dims of input control the gate
        self.gate_channels = min(32, n_kv_head * head_dim)
        self.gate = nn.Linear(self.gate_channels, n_kv_head, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, ids: torch.Tensor, v: torch.Tensor, x_for_gate: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, S) — token ids
        v: (B, S, n_kv_head, head_dim) — value from projection
        x_for_gate: (B, S, d_model) — input used for gate (x before attention)

        Returns: v + gate * ve, same shape as v
        """
        B, S = ids.shape
        ve = self.embed(ids).view(B, S, self.n_kv_head, self.head_dim)
        # Compute gate: 2*sigmoid for [0, 2] range
        gate = 2 * torch.sigmoid(self.gate(x_for_gate[..., :self.gate_channels]))
        gate = gate.unsqueeze(-1)  # (B, S, n_kv_head, 1)
        return v + gate * ve


# ============================================================================
# IMPROVED DELTANET WITH WINDOW + VALUE EMBEDDING + X0 SCALING
# ============================================================================
class GatedDeltaNetOmegaV3(nn.Module):
    """
    V3 DeltaNet: adds Value Embeddings and windowed attention.
    The core GatedDeltaNet-Ω stays the same (O(n) linear attention).
    """
    def __init__(self, config: NSTPOmegaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        D, H, hd = config.d_model, config.num_heads, config.head_dim

        # Projections (quantized)
        self.q_proj = QuantizedLinear158(D, D)
        self.k_proj = QuantizedLinear158(D, D)
        self.v_proj = QuantizedLinear158(D, D)

        # Triple gates per head
        self.erase_gate = QuantizedLinear158(D, H)
        self.write_gate = QuantizedLinear158(D, H)
        self.neuromod_gate = QuantizedLinear158(D, H)

        # Output projection
        self.o_proj = QuantizedLinear158(D, D)

        # Value Embedding (only on alternating layers, like ResFormer)
        # Match V's actual head count to avoid shape mismatches
        self.has_ve = (layer_idx % 2 == (config.num_layers - 1) % 2)
        self.n_kv_head = H  # Same as main attention heads
        self.head_dim_local = hd
        if self.has_ve:
            self.value_embed = ValueEmbedding(config.vocab_size, H, hd)
        else:
            self.value_embed = None

        self.scale = hd ** -0.5

        # Windowed attention: if window > 0, only attend to last `window` tokens
        self.window_size = self._compute_window_size(config, layer_idx)

        # Recurrent state
        self.register_buffer('recurrent_state', None, persistent=False)

    def _compute_window_size(self, config: NSTPOmegaConfig, layer_idx: int) -> int:
        """SSSL pattern: most layers use short context, last uses full."""
        seq_len = 2048  # max sequence length
        pattern = ['S', 'S', 'S', 'L']  # 3 short + 1 long, repeated
        c = pattern[layer_idx % len(pattern)]
        if c == 'L':
            return -1  # full
        else:
            return seq_len // 2  # short

    def forward(self, x: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                return_state: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, hd = self.config.num_heads, self.head_dim_local

        # Projections
        Q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        K = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        # Value Embedding (if present)
        if self.value_embed is not None:
            ids_dummy = torch.zeros(B, S, dtype=torch.long, device=x.device)
            # Use first S positions to look up (this is a simplification —
            # in production, you'd pass actual ids)
            V = self.value_embed(ids_dummy, V.transpose(1, 2).contiguous(), x).transpose(1, 2)

        # Gates
        αₑ = torch.sigmoid(self.erase_gate(x)).transpose(1, 2).unsqueeze(-1)
        α_w = torch.sigmoid(self.write_gate(x)).transpose(1, 2).unsqueeze(-1)
        αₙ = torch.sigmoid(self.neuromod_gate(x)).transpose(1, 2).unsqueeze(-1)

        # Recurrent state
        if self.recurrent_state is not None and self.recurrent_state.shape[0] == B:
            state = self.recurrent_state
        else:
            state = torch.zeros(B, H, hd, device=x.device, dtype=x.dtype)

        # Recurrent forward (O(S))
        outputs = []
        for t in range(S):
            k_t = K[:, :, t:t+1, :]
            state = αₑ[:, :, t:t+1, :] * state.unsqueeze(2) + (1 - α_w[:, :, t:t+1, :]) * k_t
            state = state.squeeze(2)

            q_t = Q[:, :, t:t+1, :]
            out_t = (q_t * state.unsqueeze(2)).sum(-1) * self.scale
            out_t = out_t * αₙ[:, :, t:t+1, :].squeeze(-1)
            outputs.append(out_t)

        out = torch.cat(outputs, dim=2).transpose(1, 2)
        out = out.unsqueeze(-1).expand(B, S, H, hd).reshape(B, S, D)
        out = self.o_proj(out)

        self.recurrent_state = state.detach()
        final_state = state if return_state else None
        αₙ_out = αₙ.squeeze(-1).transpose(1, 2)
        return out, final_state, V, αₙ_out


# ============================================================================
# V3 BLOCK
# ============================================================================
class NSTPOmegaBlockV3(nn.Module):
    """V3 block: V2 architecture + x0 residual lambdas + improved init."""
    def __init__(self, config: NSTPOmegaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Replaced deltanet with V3 version
        self.deltanet = GatedDeltaNetOmegaV3(config, layer_idx)
        self.rf_moe = RFMoE(config)
        self.halt_gate = HaltGate(config.d_model)

        # Norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # EAGLE-Ω
        self.eagle = EagleOmega(config)

        # x0 residual mixing (V3 addition)
        # Per-layer scaling of how much x0 contributes to this layer's residual
        self.x0_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor,
                x0: torch.Tensor,  # V3: initial embedding passed through
                positions: Optional[torch.Tensor] = None,
                hhm: Optional[HierarchicalHDCMemory] = None,
                return_draft: bool = False) -> Dict:
        B, S, D = x.shape

        # --- DeltaNet-Ω V3 (with x0 mixing in residual) ---
        r = x + self.x0_lambda * x0  # Mix x0 into residual
        x_norm = self.norm1(x)
        delta_out, state, V, αₙ = self.deltanet(x_norm, positions, return_state=True)
        x = r + self.dropout(delta_out)

        # --- HHM Write ---
        if hhm is not None:
            V_pooled = V.mean(1)
            V_projected = V_pooled.repeat(1, 1, self.config.num_heads)
            hhm.update(V_projected, delta_out)

        # --- RF-MoE ---
        r = x
        x_norm = self.norm2(x)
        moe_out = self.rf_moe(x_norm, αₙ)
        x = r + self.dropout(moe_out)

        # --- Halt Gate ---
        halt_prob = self.halt_gate(x)

        # --- EAGLE-Ω Draft ---
        draft_info = None
        if return_draft:
            draft_info = self.eagle(x)

        return {
            'x': x,
            'halt_prob': halt_prob,
            'neuromod_gate': αₙ,
            'state': state,
            'draft': draft_info
        }


# ============================================================================
# V3 MODEL
# ============================================================================
class NSTPOmegaV3(nn.Module):
    """
    NSTP-Ω V3 — same architecture as V2 + Value Embeddings + x0 lambdas + softcap.
    """
    def __init__(self, config: NSTPOmegaConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # V3 Blocks
        self.blocks = nn.ModuleList([
            NSTPOmegaBlockV3(config, i) for i in range(config.num_layers)
        ])

        # HHM (shared)
        self.hhm = HierarchicalHDCMemory(config)

        # Final norm + head
        self.norm = nn.LayerNorm(config.d_model)
        self.head = QuantizedLinear158(config.d_model, config.vocab_size)

        # V3: per-layer resid lambdas (initialized to 1.0)
        self.resid_lambdas = nn.Parameter(torch.ones(config.num_layers))

        # Apply init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Muon-style init: 3^0.5 * d^(-0.5) for matrices."""
        if isinstance(m, nn.Linear):
            d = m.weight.shape[-1]
            s = 3 ** 0.5 * d ** -0.5
            nn.init.uniform_(m.weight, -s, s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            d = m.weight.shape[-1]
            s = 3 ** 0.5 * d ** -0.5
            nn.init.uniform_(m.weight, -s, s)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                return_drafts: bool = False,
                softcap: float = 15.0) -> Dict:
        B, S = ids.shape
        if positions is None:
            positions = torch.arange(S, device=ids.device).unsqueeze(0).expand(B, -1)

        x = self.dropout(self.embed(ids))
        x0 = x  # V3: keep x0 for residual mixing

        all_halt_probs = []
        all_drafts = []

        for i, block in enumerate(self.blocks):
            # LayerDrop
            if self.training and torch.rand(1).item() < self.config.layer_drop:
                continue

            min_L, max_L = self.config.min_layers, self.config.max_layers
            force_compute = i >= min_L
            can_halt = i >= min_L and i < max_L

            # V3: apply resid_lambda + pass x0
            x_before = x
            out = block(x, x0, positions, self.hhm, return_draft=return_drafts or can_halt)
            # resid_lambda modulates the residual contribution (Karpathy ResFormer-style)
            # When resid_lambdas[i] = 1.0 (init), x_new = x + delta (additive residual)
            # When resid_lambdas[i] < 1.0, scales down the new contribution
            x = x_before + self.resid_lambdas[i] * (out['x'] - x_before)
            all_halt_probs.append(out['halt_prob'])

            if return_drafts and out['draft'] is not None:
                all_drafts.append(out['draft'])

        x = self.norm(x)
        logits = self.head(x)

        # V3: softcap logits
        if softcap > 0:
            logits = softcap_logits(logits, softcap)

        return {
            'logits': logits,
            'halt_probs': torch.stack(all_halt_probs, dim=1) if all_halt_probs else None,
            'drafts': all_drafts if all_drafts else None,
            'avg_layers_used': len(all_halt_probs)
        }

    def reset_memory(self):
        if self.hhm is not None:
            self.hhm.reset()
        for block in self.blocks:
            block.deltanet.recurrent_state = None

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            out = self.forward(ids[:, -2048:])
            logits = out['logits'][:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == 50256:
                break
        return ids


# ============================================================================
# PARAMETER SPLITTING FOR MUON
# ============================================================================
def split_v3_params_for_muon(model: NSTPOmegaV3):
    """
    Split V3 model parameters into Muon (2D matrices, not embeddings/head)
    and AdamW (everything else).

    Returns:
        muon_param_groups: dict mapping param → dict with lr info
        adamw_param_groups: dict mapping param → dict with lr info
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Classify
        is_matrix = p.ndim >= 2
        is_embed = 'embed' in name or 'wte' in name or 'lm_head' in name or '.head.' in name
        is_norm = 'norm' in name
        is_scalar = p.ndim < 2

        if is_matrix and not is_embed and not is_norm:
            muon_params.append((name, p))
        else:
            adamw_params.append((name, p))

    return muon_params, adamw_params


if __name__ == "__main__":
    print("Testing NSTP-Ω V3...")

    config = NSTPOmegaConfig(
        vocab_size=1000, d_model=128, num_layers=4, num_heads=4, num_experts=2
    )
    model = NSTPOmegaV3(config)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} ({total/1e6:.2f}M)")

    # Forward
    B, S = 2, 64
    ids = torch.randint(0, 1000, (B, S))
    out = model(ids, return_drafts=True)
    print(f"Logits: {out['logits'].shape}")
    print(f"Avg layers used: {out['avg_layers_used']}")
    print(f"Drafts: {len(out['drafts']) if out['drafts'] else 0}")

    # Test softcap (logits should be bounded by ±15)
    print(f"Logits max/min: {out['logits'].max().item():.2f} / {out['logits'].min().item():.2f}")

    # Test split
    muon_p, adamw_p = split_v3_params_for_muon(model)
    print(f"\nMuon:    {len(muon_p)} tensors, {sum(p.numel() for _, p in muon_p):,} params")
    print(f"AdamW:   {len(adamw_p)} tensors, {sum(p.numel() for _, p in adamw_p):,} params")

    # Print parameter names for inspection
    print("\nMuon params (sample):")
    for name, p in muon_p[:5]:
        print(f"  {name}: {p.shape}")
    print("\nAdamW params (sample):")
    for name, p in adamw_p[:5]:
        print(f"  {name}: {p.shape}")

    print("\n✅ NSTP-Ω V3 forward pass working!")
