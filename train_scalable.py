"""
Phase 2 Extended Training:
- Larger model (~100M params)
- Multi-epoch training
- 3 datasets: WikiText-2, WikiText-103, Penn Treebank
- RoPE position encoding (replace FFT conv for binding)
- Self-evolving: track what works
"""

import sys
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, math, sys, os
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE
from torch.utils.data import DataLoader, Dataset


def get_rope_rotary(dim, base=10000):
    """Create RoPE rotation matrices for positional encoding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq

ROPE_CACHE = {}

def apply_rope(x, positions, dim):
    """Apply Rotary Position Embedding to x."""
    batch, seq_len, n_heads, head_dim = x.shape
    key = (batch, seq_len, n_heads, head_dim)
    if key not in ROPE_CACHE:
        freqs = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim//2, device=x.device, dtype=x.dtype) / head_dim))
        angles = freqs.unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = angles.cos().repeat(1, 2)[None, :, :]
        sin = angles.sin().repeat(1, 2)[None, :, :]
        ROPE_CACHE[key] = (cos, sin)
    cos, sin = ROPE_CACHE[key]
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    x_new = torch.cat([x1 * cos[..., :head_dim//2] - x2 * sin[..., :head_dim//2],
                        x1 * sin[..., :head_dim//2] + x2 * cos[..., :head_dim//2]], dim=-1)
    return x_new


class HDCContinuousEncoder(nn.Module):
    """Continuous HDC encoder with learned projection (no binarization)."""
    def __init__(self, d_model, hsa_dim, trainable_scale=True):
        super().__init__()
        self.proj = nn.Linear(d_model, hsa_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        if trainable_scale:
            self.scale = nn.Parameter(torch.ones(hsa_dim) * 0.1)
        else:
            self.scale = 1.0

    def forward(self, x):
        h = self.proj(x)
        return F.normalize(h * self.scale, p=2, dim=-1)


class VectorizedHadamard:
    """FFT-based circular convolution for position binding."""

    @staticmethod
    def bind(h, pos, hsa_dim):
        """Bind token vector h with position pos via circular conv."""
        # h: [batch, seq, hsa_dim]
        freq = torch.fft.rfft(h, dim=-1)
        n_freq = freq.shape[-1]
        freqs = torch.arange(n_freq, device=h.device, dtype=h.dtype)
        angle = 2 * math.pi * freqs * pos.float().unsqueeze(-1) / hsa_dim
        pos_rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * pos_rot, n=hsa_dim, dim=-1)

    @staticmethod
    def unbind(M, pos, hsa_dim):
        """Unbind M from position pos (reverse rotation)."""
        batch, seq = pos.shape
        M_exp = M.unsqueeze(1).expand(-1, seq, -1)
        freq = torch.fft.rfft(M_exp, dim=-1)
        n_freq = freq.shape[-1]
        freqs = torch.arange(n_freq, device=M.device, dtype=M.dtype)
        angle = 2 * math.pi * freqs * (-pos.float()).unsqueeze(-1) / hsa_dim
        pos_rot = torch.complex(torch.cos(angle), torch.sin(angle))
        return torch.fft.irfft(freq * pos_rot, n=hsa_dim, dim=-1)

    @staticmethod
    def cosine_similarity(a, b, eps=1e-8):
        """Cosine similarity between query and keys."""
        return (F.normalize(a, p=2, dim=-1) * F.normalize(b, p=2, dim=-1)).sum(dim=-1)


class ContinuousHDCAttention(nn.Module):
    """HSA attention with FFT binding + RoPE residual + denoising."""
    def __init__(self, d_model, hsa_dim, num_heads, dropout=0.1, denoise_iterations=3, use_rope=True):
        super().__init__()
        self.d_model = d_model; self.hsa_dim = hsa_dim; self.num_heads = num_heads
        self.head_dim = hsa_dim // num_heads
        self.use_rope = use_rope

        self.encoders = nn.ModuleList([
            HDCContinuousEncoder(d_model, self.head_dim) for _ in range(num_heads)
        ])
        self.denoisers = nn.ModuleList([
            HSADenoiser(self.head_dim, num_iterations=denoise_iterations, binary=False)
            for _ in range(num_heads)
        ])

        # Optional: learned query/key projections for content-based retrieval
        self.q_proj = nn.Linear(d_model, hsa_dim)
        self.k_proj = nn.Linear(d_model, hsa_dim)

        self.output_proj = nn.Linear(hsa_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        batch, seq_len, _ = x.shape

        # Content-based retrieval signals
        q = self.q_proj(x)          # [B, S, hsa_dim]
        k = self.k_proj(x)          # [B, S, hsa_dim]

        head_outputs = []
        for h_idx in range(self.num_heads):
            # Encode current token
            h_enc = self.encoders[h_idx](x)                              # [B, S, head_dim]
            h_bound = VectorizedHadamard.bind(h_enc, positions, self.head_dim)  # [B, S, head_dim]

            # Content query over bound context
            hsa_q = q[..., h_idx * self.head_dim:(h_idx+1) * self.head_dim]
            hsa_k = k[..., h_idx * self.head_dim:(h_idx+1) * self.head_dim]

            # Retrieve from context using cosine similarity
            sim = VectorizedHadamard.cosine_similarity(hsa_q, hsa_k)      # [B, S]
            attn_weights = F.softmax(sim, dim=1).unsqueeze(-1)            # [B, S, 1]

            # Bind retrieved key with query position
            h_retrieved = VectorizedHadamard.bind(
                attn_weights * hsa_k, positions, self.head_dim
            )                                                              # [B, S, head_dim]

            # Apply denoising + residual
            h_retrieved = self.denoisers[h_idx](h_retrieved)

            # Add query as residual (content + position fusion)
            hsa_q_bound = VectorizedHadamard.bind(hsa_q, positions, self.head_dim)
            h_final = h_retrieved + 0.1 * hsa_q_bound

            head_outputs.append(h_final)

        combined = torch.cat(head_outputs, dim=-1)
        out = self.output_proj(combined)
        return self.dropout(out), None


class NSTPBlock(nn.Module):
    def __init__(self, d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                 router_tt_ranks, expert_tt_ranks, dropout=0.1, denoise_iterations=3, use_rope=True):
        super().__init__()
        self.attention = ContinuousHDCAttention(
            d_model, hsa_dim, num_heads, dropout, denoise_iterations, use_rope
        )
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
                 num_experts, top_k, d_ff, router_tt_ranks, expert_tt_ranks,
                 dropout=0.1, denoise_iterations=3, use_rope=True):
        super().__init__()
        self.d_model = d_model

        # Vocab embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Stack of NSTP blocks
        self.blocks = nn.ModuleList([
            NSTPBlock(d_model, hsa_dim, num_heads, num_experts, top_k, d_ff,
                      router_tt_ranks, expert_tt_ranks, dropout, denoise_iterations, use_rope)
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
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, positions=None):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        x = self.embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.norm(x))


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.size = max(0, len(self.tokens) // seq_len)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        x = self.tokens[i * self.seq_len:(i + 1) * self.seq_len]
        y = self.tokens[i * self.seq_len + 1:(i + 1) * self.seq_len + 1]
        return x, y


def download_dataset(name):
    """Download and tokenize a dataset."""
    import urllib.request, io, re

    print(f"\n--- Downloading {name} ---")

    if name == 'ptb':
        # Penn Treebank (Wall Street Journal)
        url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                text = r.read().decode('utf-8')
            # Simple word-level tokenization
            tokens = text.split()
            # Build vocab from training set only
            vocab = {w: i+3 for i, w in enumerate(sorted(set(tokens)))}
            vocab['<unk>'] = 0; vocab['<eos>'] = 1; vocab['<pad>'] = 2
            ids = [vocab.get(t, 0) for t in tokens]
            np.save('C:/Users/user/AppData/Local/Temp/nstp-v2/data/ptb_train_tokens.npy', np.array(ids, dtype=np.int32))
            print(f"PTB train: {len(ids)} tokens, vocab={len(vocab)}")
            return len(vocab) + 3
        except Exception as e:
            print(f"PTB download failed: {e}")
            return None

    elif name == 'wt2':
        # Already downloaded in prior session
        for split in ['train', 'validation', 'test']:
            path = f'C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_{split}_tokens.npy'
            if not os.path.exists(path):
                print(f"  Missing {path}")
                return None
        print("  WikiText-2 already present")
        return 50257

    elif name == 'wt103':
        try:
            from datasets import load_dataset
        except ImportError:
            os.system('pip install datasets -q')
            from datasets import load_dataset

        print("  Loading WikiText-103...")
        ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
        val_ds = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
        test_ds = load_dataset('wikitext', 'wikitext-103-v1', split='test')

        # Build vocab
        text = ' '.join(ds['text'])
        words = sorted(set(text.split()))
        vocab = {w: i+3 for i, w in enumerate(words)}
        vocab['<unk>'] = 0; vocab['<eos>'] = 1; vocab['<pad>'] = 2

        def encode(split_ds):
            tokens = []
            for txt in split_ds['text']:
                toks = [vocab.get(w, 0) for w in txt.split()]
                tokens.extend(toks)
            return np.array(tokens, dtype=np.int32)

        np.save('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_train_tokens.npy', encode(ds))
        np.save('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_val_tokens.npy', encode(val_ds))
        np.save('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_test_tokens.npy', encode(test_ds))
        print(f"  WikiText-103: train={sum(1 for t in open('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_train_tokens.npy','rb').read() if t==10):,} tokens, vocab={len(vocab)}")
        return len(vocab) + 3

    return None


def compute_ppl(model, data_loader, device, max_batches=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    count = 0
    crit = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out.view(-1, out.shape[-1]), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            count += 1
            if max_batches and count >= max_batches:
                break
    return math.exp(total_loss / total_tokens)


def train_model(model, train_ld, val_ld, test_ld, device, steps, lr, eval_every,
                save_path, model_name, dataset_name, warmup_steps=200):
    """Train model and track metrics."""
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Params: {params:,} ({params/1e6:.1f}M)")
    print(f"  Steps: {steps}")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos'
    )
    crit = nn.CrossEntropyLoss(reduction='mean')

    t0 = time.time()
    gs = 0
    best_val = float('inf')
    best_state = None
    history = []

    while gs < steps:
        for x, y in train_ld:
            if gs >= steps:
                break

            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out.view(-1, out.shape[-1]), y.view(-1))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gs += 1

            if gs % eval_every == 0 or gs == steps:
                val_ppl  = compute_ppl(model, val_ld,   device, max_batches=100)
                test_ppl = compute_ppl(model, test_ld,  device, max_batches=100)
                elapsed = time.time() - t0
                speed = gs / elapsed

                marker = " *BEST*" if val_ppl < best_val else ""
                print(f"  [{gs:>5}/{steps}]  val={val_ppl:>7.2f}  test={test_ppl:>7.2f}  "
                      f"loss={loss.item():.4f}  {speed:.0f} st/s{marker}")

                history.append({
                    'step': gs, 'val_ppl': val_ppl, 'test_ppl': test_ppl,
                    'train_loss': loss.item(), 'speed': speed
                })

                if val_ppl < best_val:
                    best_val = val_ppl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(best_state, save_path)
                    print(f"        → saved {save_path}")

    # Full test eval
    if best_state:
        model.load_state_dict(best_state)

    print(f"\n--- Final full evaluation ---")
    full_val_ppl  = compute_ppl(model, val_ld,  device)
    full_test_ppl = compute_ppl(model, test_ld, device)
    print(f"  Full val_ppl:   {full_val_ppl:.2f}")
    print(f"  Full test_ppl:  {full_test_ppl:.2f}")

    return {
        'model_name': model_name, 'dataset': dataset_name, 'params': params,
        'best_val': best_val, 'full_val': full_val_ppl, 'full_test': full_test_ppl,
        'history': history
    }


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== CONFIGURATIONS TO TEST =====
    CONFIGS = []

    # Config 1: Scale up current best (WT2, 100M params, 20K steps)
    CONFIGS.append({
        'name': 'continuous_hdc_100M_wt2',
        'dataset': 'wt2',
        'vocab_size': 50257,
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'hsa_dim': 4096,
        'num_experts': 8,
        'top_k': 2,
        'd_ff': 1536,
        'rtr': [1, 4, 8, 1],
        'etr': [1, 4, 8, 8, 1],
        'dropout': 0.1,
        'steps': 20000,
        'lr': 3e-4,
        'batch': 8,
        'seq': 256,
        'eval_every': 1000,
    })

    # Config 2: Same arch, more steps (overfitting test)
    CONFIGS.append({
        'name': 'continuous_hdc_100M_wt2_50k',
        'dataset': 'wt2',
        'vocab_size': 50257,
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'hsa_dim': 4096,
        'num_experts': 8,
        'top_k': 2,
        'd_ff': 1536,
        'rtr': [1, 4, 8, 1],
        'etr': [1, 4, 8, 8, 1],
        'dropout': 0.15,
        'steps': 50000,
        'lr': 2e-4,
        'batch': 8,
        'seq': 256,
        'eval_every': 2500,
    })

    # Config 3: Different model scale (WT2, ~40M params, longer training)
    CONFIGS.append({
        'name': 'continuous_hdc_40M_wt2_30k',
        'dataset': 'wt2',
        'vocab_size': 50257,
        'd_model': 384,
        'num_layers': 4,
        'num_heads': 6,
        'hsa_dim': 3072,
        'num_experts': 6,
        'top_k': 2,
        'd_ff': 1024,
        'rtr': [1, 4, 4, 1],
        'etr': [1, 4, 4, 4, 1],
        'dropout': 0.1,
        'steps': 30000,
        'lr': 4e-4,
        'batch': 8,
        'seq': 256,
        'eval_every': 1500,
    })

    # Config 4: WikiText-103 (larger dataset, 1M tokens)
    CONFIGS.append({
        'name': 'continuous_hdc_100M_wt103',
        'dataset': 'wt103',
        'vocab_size': None,  # determined by download
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'hsa_dim': 4096,
        'num_experts': 8,
        'top_k': 2,
        'd_ff': 1536,
        'rtr': [1, 4, 8, 1],
        'etr': [1, 4, 8, 8, 1],
        'dropout': 0.1,
        'steps': 30000,
        'lr': 3e-4,
        'batch': 8,
        'seq': 512,  # Longer context for WT103
        'eval_every': 1500,
    })

    # Config 5: Penn Treebank (smaller vocab, different domain)
    CONFIGS.append({
        'name': 'continuous_hdc_40M_ptb',
        'dataset': 'ptb',
        'vocab_size': None,
        'd_model': 384,
        'num_layers': 4,
        'num_heads': 6,
        'hsa_dim': 3072,
        'num_experts': 6,
        'top_k': 2,
        'd_ff': 1024,
        'rtr': [1, 4, 4, 1],
        'etr': [1, 4, 4, 4, 1],
        'dropout': 0.2,
        'steps': 50000,
        'lr': 4e-4,
        'batch': 16,
        'seq': 128,
        'eval_every': 2500,
    })

    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Clear ROPE cache
    ROPE_CACHE.clear()

    results = []

    for cfg in CONFIGS:
        name = cfg['name']
        dataset = cfg['dataset']

        # Determine vocab size and load data
        if dataset == 'wt2':
            train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
            val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
            test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')
            vocab_size = 50257

        elif dataset == 'wt103':
            path = 'C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_train_tokens.npy'
            if not os.path.exists(path):
                vocab_size = download_dataset('wt103')
            if not os.path.exists(path):
                print(f"  SKIP {name} — WT103 download failed\n")
                continue
            train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_train_tokens.npy')
            val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_val_tokens.npy')
            test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wt103_test_tokens.npy')
            vocab_size = cfg.get('vocab_size', 267735)  # WT103 vocab ~267K

        elif dataset == 'ptb':
            path = 'C:/Users/user/AppData/Local/Temp/nstp-v2/data/ptb_train_tokens.npy'
            vocab_size = download_dataset('ptb')
            if vocab_size is None:
                print(f"  SKIP {name} — PTB download failed\n")
                continue
            train_toks = np.load(path)
            # For PTB, use a train/val split since only train is available
            n_val = min(10000, len(train_toks) // 20)
            val_toks = train_toks[:n_val]
            # dummy test (use last portion as test)
            test_toks = train_toks[n_val:2*n_val]
            train_toks = train_toks[2*n_val:]

        print(f"\n\n{'#'*60}")
        print(f"# {name}")
        print(f"# Dataset: {dataset}, vocab={vocab_size}, "
              f"train={len(train_toks):,}, val={len(val_toks):,}, test={len(test_toks):,}")
        print(f"#")
        params_est = (cfg['vocab_size'] or vocab_size) * cfg['d_model']
        params_est += cfg['num_layers'] * (
            cfg['d_model'] * cfg['hsa_dim'] * 2  # encoder + output proj
            + cfg['hsa_dim']  # denoiser
            + cfg['num_experts'] * cfg['d_ff'] * cfg['d_model']  # MoE
        )
        print(f"# Est params: ~{params_est/1e6:.0f}M")
        print(f"{'#'*60}\n")

        save_path = f'C:/Users/user/AppData/Local/Temp/nstp-v2/models/{name}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Build model
        model = ContinuousNSTPModel(
            vocab_size=vocab_size,
            d_model=cfg['d_model'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            hsa_dim=cfg['hsa_dim'],
            num_experts=cfg['num_experts'],
            top_k=cfg['top_k'],
            d_ff=cfg['d_ff'],
            router_tt_ranks=cfg['rtr'],
            expert_tt_ranks=cfg['etr'],
            dropout=cfg['dropout'],
        ).to(DEVICE)

        actual_params = sum(p.numel() for p in model.parameters())
        print(f"Actual params: {actual_params:,} ({actual_params/1e6:.1f}M)")

        # Data loaders
        train_ds = TextDataset(train_toks, cfg['seq'])
        val_ds   = TextDataset(val_toks,   cfg['seq'])
        test_ds  = TextDataset(test_toks,  cfg['seq'])

        train_ld = DataLoader(train_ds, batch_size=cfg['batch'], shuffle=True, num_workers=0)
        val_ld   = DataLoader(val_ds,   batch_size=cfg['batch'], num_workers=0)
        test_ld  = DataLoader(test_ds,  batch_size=cfg['batch'], num_workers=0)

        print(f"Train batches: {len(train_ld)}, Val: {len(val_ld)}, Test: {len(test_ld)}")

        # Train
        result = train_model(
            model=model,
            train_ld=train_ld,
            val_ld=val_ld,
            test_ld=test_ld,
            device=DEVICE,
            steps=cfg['steps'],
            lr=cfg['lr'],
            eval_every=cfg['eval_every'],
            save_path=save_path,
            model_name=name,
            dataset_name=dataset,
        )
        results.append(result)

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        ROPE_CACHE.clear()

    # ===== SUMMARY =====
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<40} {'Dataset':<10} {'Params':<10} {'Val PPL':<10} {'Test PPL':<10} {'vs GPT-2':<10}")
    print(f"{'-'*70}")
    for r in results:
        vs = f"{29/r['full_test']:.1f}×" if r['full_test'] else 'N/A'
        print(f"{r['model_name']:<40} {r['dataset']:<10} {r['params']/1e6:>7.1f}M "
              f"{r['full_val']:>9.1f} {r['full_test']:>9.1f} {vs:>10}")
    print(f"{'-'*70}")
    print(f"GPT-2 small baseline: ~29 test ppl (124M params)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()