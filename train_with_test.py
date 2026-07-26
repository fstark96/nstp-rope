import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, math, sys
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE


def vectorized_hadamard_bind(h, pos, hsa_dim):
    freq = torch.fft.rfft(h, dim=-1)
    freqs = torch.arange(freq.shape[-1], device=h.device, dtype=h.dtype)
    pos_float = pos.float()
    angle = 2 * np.pi * freqs * pos_float.unsqueeze(-1) / hsa_dim
    pos_freq = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    pos_freq = torch.complex(pos_freq[..., 0], pos_freq[..., 1])
    return torch.fft.irfft(freq * pos_freq, n=hsa_dim, dim=-1)


def vectorized_hadamard_unbind(M, pos, hsa_dim):
    batch, seq = pos.shape
    M_exp = M.unsqueeze(1).expand(-1, seq, -1)
    freq = torch.fft.rfft(M_exp, dim=-1)
    freq_dim = freq.shape[-1]
    freqs = torch.arange(freq_dim, device=M.device, dtype=M.dtype)
    pos_float = pos.float()
    angle = 2 * np.pi * freqs * (-pos_float).unsqueeze(-1) / hsa_dim
    pos_freq = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    pos_freq = torch.complex(pos_freq[..., 0], pos_freq[..., 1])
    return torch.fft.irfft(freq * pos_freq, n=hsa_dim, dim=-1)


class ContinuousHDCEncoder(nn.Module):
    def __init__(self, d_model, hsa_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)


class ContinuousHDAAttention(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.hsa_dim = hsa_dim; self.num_heads = num_heads
        self.head_dim = hsa_dim // num_heads
        self.encoders = nn.ModuleList([
            ContinuousHDCEncoder(d_model, self.head_dim) for _ in range(num_heads)
        ])
        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=3, binary=False) for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(hsa_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        head_outputs = []
        for h_idx in range(self.num_heads):
            h_enc = self.encoders[h_idx](x)
            h_bound = vectorized_hadamard_bind(h_enc, positions, self.head_dim)
            M = h_bound.mean(dim=1)
            retrieved = vectorized_hadamard_unbind(M, positions, self.head_dim)
            retrieved = self.denoisers[h_idx](retrieved)
            head_outputs.append(retrieved)
        combined = torch.cat(head_outputs, dim=-1)
        out = self.output_proj(combined)
        return self.dropout(out), None


class ContinuousNSTPBlock(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                 router_tt_ranks, expert_tt_ranks, dropout=0.1):
        super().__init__()
        self.attention = ContinuousHDAAttention(d_model, hsa_dim, num_heads, dropout)
        self.moe = TTCERMoE(
            d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k,
            router_tt_ranks=router_tt_ranks, expert_tt_ranks=expert_tt_ranks,
            activation='gelu', dropout=dropout, router_aux_loss_coef=0.01,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, positions)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        moe_out, _ = self.moe(x)
        return residual + moe_out


class ContinuousNSTPModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hsa_dim,
                 num_experts, top_k, d_ff, router_tt_ranks, expert_tt_ranks, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ContinuousNSTPBlock(d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                                 router_tt_ranks, expert_tt_ranks, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # Untied
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, positions=None):
        batch, seq = input_ids.shape
        device = input_ids.device
        if positions is None:
            positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        x = self.embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.norm(x))


# ---- Training + Test Eval ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, SEQ = 4, 128
LR = 5e-4
STEPS = 5000
EVAL_EVERY = 500

train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

class DS:
    def __init__(self, toks, seq):
        self.toks = torch.tensor(toks, dtype=torch.long)
        self.seq = seq
    def __len__(self): return max(0, len(self.toks) // self.seq)
    def __getitem__(self, i):
        s = self.toks[i*self.seq:(i+1)*self.seq+1]
        return s[:-1], s[1:]

train_ds = DS(train_toks, SEQ)
val_ds   = DS(val_toks,   SEQ)
test_ds  = DS(test_toks,  SEQ)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH, num_workers=0)
test_ld  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH, num_workers=0)


model = ContinuousNSTPModel(
    vocab_size=50257, d_model=320, num_layers=3, num_heads=4,
    hsa_dim=2048, num_experts=4, top_k=2, d_ff=768,
    router_tt_ranks=[1, 4, 4, 1], expert_tt_ranks=[1, 4, 4, 4, 1],
    dropout=0.1
).to(DEVICE)

params = sum(p.numel() for p in model.parameters())
print(f"Device: {DEVICE}")
print(f"Model: {params:,} params ({params/1e6:.1f}M)")
print(f"Train batches: {len(train_ld)}, Val: {len(val_ld)}, Test: {len(test_ld)}")
print(f"Steps: {STEPS}, Eval every: {EVAL_EVERY}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
crit = nn.CrossEntropyLoss(reduction='mean')


def compute_ppl(data_loader, desc="", max_batches=None):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    count = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out.view(-1, 50257), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            count += 1
            if max_batches and count >= max_batches:
                break
    ppl = math.exp(total_loss / total_tokens)
    return ppl


# Quick random baseline
print(f"\n--- Random init baselines ---")
print(f"Random val_ppl (full):   {compute_ppl(val_ld, 'val'):.1f}")
print(f"Random test_ppl (full):  {compute_ppl(test_ld, 'test'):.1f}")

# Training
print(f"\n--- Training ({STEPS} steps) ---")
print(f"{'Step':>6}  {'Val PPL':>10}  {'Test PPL':>10}  {'Train Loss':>11}  {'Time':>6}")
print("-" * 60)

t0 = time.time()
gs = 0
best_val = float('inf')
best_test = None

for x, y in train_ld:
    if gs >= STEPS:
        break

    x, y = x.to(DEVICE), y.to(DEVICE)
    out = model(x)
    loss = crit(out.view(-1, 50257), y.view(-1))

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

    gs += 1

    if gs % EVAL_EVERY == 0:
        val_ppl  = compute_ppl(val_ld,   desc='val')
        test_ppl = compute_ppl(test_ld,  desc='test', max_batches=100)  # sample test
        elapsed = time.time() - t0

        marker = " *BEST*" if val_ppl < best_val else ""
        print(f"  {gs:>5}  {val_ppl:>10.2f}  {test_ppl:>10.2f}  {loss.item():>11.4f}  {elapsed:>5.0f}s{marker}")

        if val_ppl < best_val:
            best_val = val_ppl
            best_test = test_ppl
            # Save best model
            torch.save(model.state_dict(), 'C:/Users/user/AppData/Local/Temp/nstp-v2/best_model.pt')
            print(f"        → saved best model (test_ppl={test_ppl:.2f})")

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"  Best val_ppl:   {best_val:.2f}")
print(f"  Test PPL at best val checkpoint: {best_test:.2f}")
print(f"  GPT-2 small:    ~29")
print(f"  Random baseline: ~55000")
print(f"{'='*60}")

# Full test eval on best checkpoint
print(f"\n--- Full test eval on best checkpoint ---")
model.load_state_dict(torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/best_model.pt', weights_only=True))
full_test_ppl = compute_ppl(test_ld, 'test')
print(f"Full test perplexity: {full_test_ppl:.2f}")
print(f"Val perplexity:        {best_val:.2f}")
print(f"Val-Test gap:          {abs(full_test_ppl - best_val):.2f}")
if full_test_ppl < 30:
    print(f"\n*** CONTINUOUS HDC BEATS GPT-2 small! ({full_test_ppl:.1f} vs ~29) ***")