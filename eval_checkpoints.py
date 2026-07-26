"""
Quick eval: measure model quality on WikiText-2 test set.
Works with any saved checkpoint.
"""
import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, sys, os
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ = 256

# Tokenized data
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

def make_loader(toks, seq, batch=8, shuffle=False):
    N = len(toks) // seq
    x = torch.tensor(toks[:N*seq], dtype=torch.long).view(N, seq)
    y = torch.tensor(toks[1:N*seq+1], dtype=torch.long).view(N, seq)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)

val_ld  = make_loader(val_toks, SEQ, batch=8)
test_ld = make_loader(test_toks, SEQ, batch=8)

def compute_ppl(model, loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    crit = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out.view(-1, 50257), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    return math.exp(total_loss / total_tokens)

print(f"Device: {DEVICE}")
print(f"Test batches: {len(test_ld)}, Val batches: {len(val_ld)}")


class VH:
    @staticmethod
    def bind(h, pos, hsa_dim):
        freq = torch.fft.rfft(h, dim=-1)
        n_freq = freq.shape[-1]
        freqs = torch.arange(n_freq, device=h.device, dtype=h.dtype)
        angle = 2 * math.pi * freqs * pos.float().unsqueeze(-1) / hsa_dim
        pos_rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * pos_rot, n=hsa_dim, dim=-1)

    @staticmethod
    def unbind(M, pos, hsa_dim):
        batch, seq = pos.shape
        M_exp = M.unsqueeze(1).expand(-1, seq, -1)
        freq = torch.fft.rfft(M_exp, dim=-1)
        n_freq = freq.shape[-1]
        freqs = torch.arange(n_freq, device=M.device, dtype=M.dtype)
        angle = 2 * math.pi * freqs * (-pos.float()).unsqueeze(-1) / hsa_dim
        pos_rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * pos_rot, n=hsa_dim, dim=-1)

    @staticmethod
    def cosim(a, b):
        return (F.normalize(a, p=2, dim=-1) * F.normalize(b, p=2, dim=-1)).sum(dim=-1)


class HDCEnc(nn.Module):
    def __init__(self, d_model, hsa_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        self.scale = nn.Parameter(torch.ones(hsa_dim) * 0.1)

    def forward(self, x):
        return F.normalize(self.proj(x) * self.scale, p=2, dim=-1)


class HDAttn(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1, n_iter=3):
        super().__init__()
        self.d_model = d_model
        self.hsa_dim = hsa_dim
        self.num_heads = num_heads
        self.head_dim = hsa_dim // num_heads

        self.encoders = nn.ModuleList([
            HDCEnc(d_model, self.head_dim) for _ in range(num_heads)
        ])
        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=n_iter, binary=False)
            for _ in range(num_heads)
        ])
        self.q_proj = nn.Linear(d_model, hsa_dim)
        self.k_proj = nn.Linear(d_model, hsa_dim)
        self.out_proj = nn.Linear(hsa_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        heads = []

        for h in range(self.num_heads):
            h_enc = self.encoders[h](x)
            h_bound = VH.bind(h_enc, positions, self.head_dim)

            sq = VH.cosim(
                q[..., h*self.head_dim:(h+1)*self.head_dim],
                k[..., h*self.head_dim:(h+1)*self.head_dim]
            )
            w = F.softmax(sq, dim=1).unsqueeze(-1)
            h_ret = VH.bind(
                w * k[..., h*self.head_dim:(h+1)*self.head_dim],
                positions, self.head_dim
            )
            h_ret = self.denoisers[h](h_ret)

            sq_bound = VH.bind(
                q[..., h*self.head_dim:(h+1)*self.head_dim],
                positions, self.head_dim
            )
            h_ret = h_ret + 0.1 * sq_bound
            heads.append(h_ret)

        out = self.out_proj(torch.cat(heads, dim=-1))
        return self.dropout(out), None


class NSTPBlock(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                 router_tt_ranks, expert_tt_ranks, dropout=0.1):
        super().__init__()
        self.attn = HDAttn(d_model, hsa_dim, num_heads, dropout)
        self.moe = TTCERMoE(
            d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
            router_tt_ranks=router_tt_ranks, expert_tt_ranks=expert_tt_ranks,
            activation='gelu', dropout=dropout, router_aux_loss_coef=0.01,
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, positions):
        r = x; x = self.n1(x); a, _ = self.attn(x, positions); x = r + self.drop(a)
        r = x; x = self.n2(x); m, _ = self.moe(x); return r + m


class NSTPModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hsa_dim,
                 num_experts, top_k, d_ff, router_tt_ranks, expert_tt_ranks,
                 dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            NSTPBlock(d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                      router_tt_ranks, expert_tt_ranks, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids, positions=None):
        B, S = ids.shape
        dev = ids.device
        if positions is None:
            positions = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.drop(self.embed(ids))
        for b in self.blocks:
            x = b(x, positions)
        return self.head(self.norm(x))


def infer_and_eval(fp):
    """Load a checkpoint, infer config, evaluate."""
    sd = torch.load(fp, map_location=DEVICE, weights_only=True)

    # Count blocks
    n_blocks = 0
    for k in sorted(sd.keys()):
        if k.startswith('blocks.') and '.attn.' in k:
            parts = k.split('.')
            idx = int(parts[1])
            n_blocks = max(n_blocks, idx + 1)

    if n_blocks == 0:
        return None, "no blocks found"

    # Vocab size from embed weight
    vs = sd['embed.weight'].shape[0]
    dm = sd['embed.weight'].shape[1]

    # Detect head_dim from q_proj weight
    q_w = sd.get('blocks.0.attn.q_proj.weight', None)
    if q_w is not None:
        hd = q_w.shape[0]
    else:
        hd = 2048

    # Detect num_heads
    hsa_dim_attr = sd.get('blocks.0.attn.hsa_dim', None)
    num_heads = sd.get('blocks.0.attn.num_heads', 4)
    if hsa_dim_attr is not None:
        hd = int(hsa_dim_attr)
    head_dim = hd // num_heads if num_heads > 0 else 256

    # Detect num_experts
    ne = 8
    for k in sd:
        if 'moe.gate.weight' in k:
            ne = sd[k].shape[0]
            break

    # Default ranks
    rtr = [1, 4, 8, 1]
    etr = [1, 4, 8, 8, 1]

    print(f"  Inferring: {n_blocks}L, d={dm}, hsa={hd}, heads={num_heads}, experts={ne}")

    model = NSTPModel(
        vs, dm, n_blocks, num_heads, hd, ne, 2, 1536, rtr, etr
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}")

    model.to(DEVICE)
    model.eval()

    val_ppl  = compute_ppl(model, val_ld)
    test_ppl = compute_ppl(model, test_ld)

    del model
    torch.cuda.empty_cache()

    return {'val': val_ppl, 'test': test_ppl}, None


# Check saved checkpoints
model_dir = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models'
results = []

print(f"\n--- Checking checkpoints in {model_dir} ---")

for fn in sorted(os.listdir(model_dir)):
    if not fn.endswith('.pt'):
        continue
    fp = os.path.join(model_dir, fn)
    sz = os.path.getsize(fp) / 1e6
    print(f"\n{'='*50}")
    print(f"Checkpoint: {fn} ({sz:.0f} MB)")

    try:
        result, err = infer_and_eval(fp)
        if err:
            print(f"  ERROR: {err}")
            continue

        params_est = sum(1 for _ in torch.load(fp, map_location='cpu', weights_only=True).values())
        print(f"  Val PPL:  {result['val']:.2f}")
        print(f"  Test PPL: {result['test']:.2f}")
        print(f"  vs GPT-2 small (~29): {29/result['test']:.2f}x better")
        results.append({'fn': fn, **result})

        if result['test'] < 29:
            print(f"  *** BEATS GPT-2 small! ***")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*50}")
print("Summary:")
print(f"  GPT-2 small baseline: ~29 test ppl (124M params)")
print(f"  Best prior run: test_ppl=7.59 (39.3M params, 1 epoch)")
for r in results:
    print(f"  {r['fn']}: val={r['val']:.1f}, test={r['test']:.1f}")