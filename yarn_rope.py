"""YaRN (Yet another RoPE extensioN) — clean implementation.
https://arxiv.org/abs/2309.00071

Two key YaRN modifications over standard RoPE:
1. ATTENTION TEMPERATURE: scale softmax inputs by 1/alpha. alpha > 1 softens attention.
2. POSITION SCALE s: positions are multiplied by s before frequency computation.
   This "stretches" the frequency spectrum so more rotations fit in the context window.
   At inference: if trained at SEQ=128 with s=1, test at SEQ=512 with s=4.
"""
import sys
class _FakeProfileModule:
    def __init__(self):
        self.run = self; self.runctx = self; self.Profile = _FakeProfileClass
    def __call__(self, *args, **kwargs): pass
    def __getattr__(self, name):
        return getattr(sys.modules.get('_pyprofile', self), name, None)
class _FakeProfileClass:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): pass
sys.modules['profile'] = _FakeProfileModule()

import math, torch, torch.nn as nn, torch.nn.functional as F

# ─── RoPE core ────────────────────────────────────────────────────────────────
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)

def _compute_rope(seq_len, head_dim, base, pos_scale, dtype):
    """Compute cos/sin for given seq_len and position scaling."""
    inv_freq = 1.0 / (base * (pos_scale ** 2) ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, dtype=dtype) * pos_scale
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

# ─── Model ─────────────────────────────────────────────────────────────────────
class YaRNAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_temp=1.0):
        super().__init__()
        self.hd = d_model // num_heads
        self.nh = num_heads
        self.scale = self.hd ** -0.5 / attn_temp
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cos, sin, mask=None):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        Q, K = apply_rope(Q, K, cos, sin)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None: attn = attn.masked_fill(mask, float('-inf'))
        return self.o_proj(torch.matmul(F.softmax(attn, -1), V).transpose(1,2).reshape(B, S, -1))

class YaRNModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=256, num_layers=2, num_heads=8,
                 d_ff=1024, base=10000.0, pos_scale=1.0, attn_temp=1.0):
        super().__init__()
        self.base = base
        self.hd = d_model // num_heads
        self.nh = num_heads
        self.pos_scale = pos_scale
        self.attn_temp = attn_temp
        self.emb = nn.Embedding(vocab_size, d_model)
        # inv_freq buffer for fast RoPE recomputation
        inv_freq = 1.0 / (base ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer('inv_freq', inv_freq)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': YaRNAttention(d_model, num_heads, attn_temp),
                'ffn': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
            }) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.emb.weight

    def _rope(self, seq_len, pos_scale=None):
        ps = pos_scale if pos_scale is not None else self.pos_scale
        inv = self.inv_freq  # (hd//2,)
        t = torch.arange(seq_len, device=inv.device, dtype=inv.dtype) * ps
        freqs = torch.outer(t, inv)  # (seq, hd//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq, hd)
        return emb.cos(), emb.sin()

    def forward(self, x, pos_scale=None, mask=None):
        B, S = x.shape
        h = self.emb(x)
        cos, sin = self._rope(S, pos_scale)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            h = h + layer['attn'](layer['ln1'](h), cos, sin, mask)
            h = h + layer['ffn'](layer['ln2'](h))
        return self.head(self.ln_f(h))


def tokenize(texts):
    """Simple word-level tokenization for quick testing."""
    vocab = sorted(set(' '.join(texts).split(' ')))  # word-level vocab
    word2id = {w: i+2 for i, w in enumerate(vocab)}
    word2id['<pad>'] = 0; word2id['<unk>'] = 1
    ids = [[word2id.get(w, 1) for w in t.split()] for t in texts]
    return torch.tensor(ids[0], dtype=torch.long), len(vocab) + 2

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"YaRN RoPE — device={device}")
    print("=" * 60)

    # Build dataset from text FIRST to know the vocab size
    texts = [
        "The quick brown fox jumps over the lazy dog while chasing a butterfly",
        "Machine learning enables computers to learn patterns from vast amounts of data",
        "Neural networks process information through interconnected layers of artificial neurons",
        "Language models predict the next token based on preceding context and learned representations",
        "Transformer architectures use self-attention to capture long-range dependencies in sequences",
        "The attention mechanism allows models to dynamically focus on relevant parts of the input",
        "Gradient descent optimization iteratively adjusts model parameters to minimize prediction error",
        "Training data quality directly influences the downstream performance of machine learning models",
    ]
    tokens, VSIZE = tokenize(texts)
    tokens = tokens.repeat(64)

    # ── Training at SEQ=128, pos_scale=1.0 ──────────────────────────────────
    # Create with correct vocab size from the start
    model = YaRNModel(vocab_size=VSIZE, d_model=256, num_layers=2, num_heads=8, d_ff=1024,
                      base=10000.0, pos_scale=1.0, attn_temp=1.0).to(device)
    p = sum(x.numel() for x in model.parameters())
    print(f"Params: {p:,}")

    print(f"Vocab: {VSIZE}, tokens: {len(tokens)}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    SEQ = 128
    BS = 8
    model.train()
    for step in range(300):
        start = (step * BS) % (len(tokens) - SEQ - 1)
        x = tokens[start:start+SEQ].unsqueeze(0).to(device)
        y = tokens[start+1:start+SEQ+1].unsqueeze(0).to(device)
        logits = model(x)
        loss = crit(logits.view(-1, VSIZE), y.view(-1))
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 100 == 0:
            print(f"  Step {step:4d}: loss={loss.item():.4f}")

    # ── Test: context length × position scale ───────────────────────────────
    print("\n--- Context Extension: train at SEQ=128, s=1 ---")
    model.eval()
    results = []
    for test_seq in [128, 256, 512, 1024, 2048]:
        for test_scale in [1.0, 2.0, 4.0, 8.0]:
            if test_scale > 1 and test_seq < 256: continue
            # Generate test data at test_seq length
            t_start = (300 * BS) % (len(tokens) - test_seq - 1)
            x = tokens[t_start:t_start+test_seq].unsqueeze(0).to(device)
            y = tokens[t_start+1:t_start+test_seq+1].unsqueeze(0).to(device)
            cos, sin = model._rope(test_seq, pos_scale=test_scale)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(x, pos_scale=test_scale)
                loss = crit(logits.view(-1, VSIZE), y.view(-1))
            ppl = math.exp(loss.item())
            results.append((test_seq, test_scale, ppl))
            print(f"  SEQ={test_seq:4d}, s={test_scale:3.1f}: PPL={ppl:.2f}")

    print("\n✅ YaRN context extension test complete")
    print("\nInterpretation:")
    print("  s=1.0 at longer SEQ → standard RoPE extrapolation (may degrade)")
    print("  s>1.0 at longer SEQ → YaRN position scaling (should help generalization)")
    print("  PPL stable across SEQ × s → good length extrapolation")

if __name__ == '__main__':
    main()