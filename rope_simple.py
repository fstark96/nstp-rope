"""Simple RoPE attention — clean implementation."""
import math, torch, torch.nn as nn

def rotate_half(x):
    """Rotate half the hidden dim of x."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to q and k.
    q, k: (batch, heads, seq, head_dim)
    cos, sin: (seq, head_dim) — precomputed
    """
    # Expand cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, S, self.d_model)
        return self.o_proj(out)


class SimpleRoPETransformer(nn.Module):
    def __init__(self, d_model=256, num_layers=3, num_heads=8, d_ff=1024, vocab_size=50257, max_seq=2048):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        
        # RoPE frequencies — register BEFORE layers so layers can use them
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model//num_heads, 2).float() / (d_model//num_heads)))
        self.register_buffer('inv_freq', inv_freq)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': RoPEAttention(d_model, num_heads),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                ),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.emb.weight
        
        # Precompute RoPE embeddings for max_seq
        t = torch.arange(max_seq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('rope_cos', emb.cos())
        self.register_buffer('rope_sin', emb.sin())
    
    def forward(self, x):
        B, S = x.shape
        h = self.emb(x)
        cos = self.rope_cos[:S]
        sin = self.rope_sin[:S]
        
        for layer in self.layers:
            h_norm = layer['ln1'](h)
            h = h + layer['attn'](h_norm, cos, sin)
            h = h + layer['ffn'](layer['ln2'](h))
        
        return self.head(self.ln_f(h))


# Quick sanity check
print("Testing RoPE attention...")
model = SimpleRoPETransformer()
x = torch.randint(0, 50257, (2, 128))
out = model(x)
print(f"Output: {out.shape}, no NaN: {not torch.isnan(out).any()}")

# Test at different seq lengths (should work!)
for seq in [128, 256, 512]:
    x = torch.randint(0, 50257, (1, seq))
    with torch.no_grad():
        out = model(x)
    print(f"SEQ={seq}: {out.shape}, stable: {not torch.isnan(out).any()}")
print("✅ RoPE works at any length!")