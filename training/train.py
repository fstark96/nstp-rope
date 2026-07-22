"""
NSTP Training Script
Trains NSTP model on language modeling task with all auxiliary losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import math
import os
from typing import Dict, Optional, List
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.insert(0, '/c/Users/online/NSTP')

from nstp_core import NSTPModel, NSTPConfig, NSTPLoss


@dataclass
class TrainingConfig:
    # Model
    model_config: NSTPConfig = None
    
    # Data
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    seq_len: int = 1024
    batch_size: int = 8
    num_workers: int = 4
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduling
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_decay: str = "cosine"  # "cosine", "linear", "constant"
    min_lr: float = 3e-5
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "checkpoints"
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient accumulation
    grad_accum_steps: int = 1
    
    # HSA specific
    orthogonalize_every: int = 1000  # Orthogonalize TT-cores periodically


class TextDataset(Dataset):
    """Simple dataset for language modeling from tokenized binary files."""
    
    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        # Load as memory-mapped for large files
        self.data = torch.from_numpy(
            np.memmap(data_path, dtype=np.uint16, mode='r')
        ).long()
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + 1 + self.seq_len]
        return x, y


def create_optimizer(model: nn.Module, config: TrainingConfig):
    """Create optimizer with weight decay only on non-norm parameters."""
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith('bias') or isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                no_decay.add(fpn)
            elif pn.endswith('weight'):
                decay.add(fpn)
    
    # Verify
    inter = decay & no_decay
    assert len(inter) == 0
    
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    return optimizer


def get_lr_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler."""
    if config.lr_decay == "cosine":
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    elif config.lr_decay == "linear":
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return max(config.min_lr / config.learning_rate, 1 - (step - config.warmup_steps) / (config.max_steps - config.warmup_steps))
    else:
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model: NSTPModel, dataloader: DataLoader, device: str, max_batches: int = 50) -> Dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            
            logits, aux_losses = model(x, return_aux_losses=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return {
        'val_loss': avg_loss,
        'val_ppl': ppl,
    }


def train_nstp(config: TrainingConfig):
    """Main training loop."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    # Create model
    if config.model_config is None:
        config.model_config = NSTPConfig()
    
    model = NSTPModel(config.model_config).to(device)
    
    # Loss function
    loss_fn = NSTPLoss(
        vocab_size=config.model_config.vocab_size,
        d_model=config.model_config.d_model,
        hsa_dim=config.model_config.hsa_dim,
        num_experts=config.model_config.num_experts,
        num_layers=config.model_config.num_layers,
        hsa_denoise_coef=config.model_config.hsa_denoise_loss_coef,
        tt_ortho_coef=config.model_config.tt_ortho_loss_coef,
        router_balance_coef=config.model_config.router_aux_loss_coef,
    )
    
    # Optimizer & scheduler
    optimizer = create_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config)
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device == 'cuda' else None
    
    # Data loaders (placeholder - replace with actual data)
    # train_dataset = TextDataset(config.train_data_path, config.seq_len)
    # val_dataset = TextDataset(config.val_data_path, config.seq_len)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    # For testing, use random data
    def get_batch():
        x = torch.randint(0, config.model_config.vocab_size, (config.batch_size, config.seq_len), device=device)
        y = torch.randint(0, config.model_config.vocab_size, (config.batch_size, config.seq_len), device=device)
        return x, y
    
    # Training loop
    model.train()
    step = 0
    best_val_ppl = float('inf')
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Starting training for {config.max_steps} steps...")
    print(f"Model params: {model.num_parameters():,}")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    
    pbar = tqdm(total=config.max_steps, desc="Training")
    
    while step < config.max_steps:
        # Get batch
        x, y = get_batch()
        
        # Forward with AMP
        if config.use_amp and device == 'cuda':
            with torch.cuda.amp.autocast():
                logits, aux_losses = model(x, return_aux_losses=True)
                
                # Compute loss
                loss_dict = loss_fn(
                    logits, y, aux_losses,
                    # HSA denoising would need retrieved/target hypervectors
                    # Skipping for now - add when HSA encoder is integrated
                )
                loss = loss_dict['total'] / config.grad_accum_steps
        else:
            logits, aux_losses = model(x, return_aux_losses=True)
            loss_dict = loss_fn(logits, y, aux_losses)
            loss = loss_dict['total'] / config.grad_accum_steps
        
        # Backward
        if config.use_amp and device == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.grad_accum_steps == 0:
            if config.use_amp and device == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
        
        # Periodic TT-core orthogonalization
        if step > 0 and step % config.orthogonalize_every == 0:
            model.orthogonalize_all_cores()
            print(f"\nStep {step}: Orthogonalized TT-cores")
        
        # Logging
        if step % config.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'ce': f"{loss_dict.get('ce', 0):.4f}",
                'moe': f"{loss_dict.get('moe_balance', 0):.4f}",
                'tt': f"{loss_dict.get('tt_ortho', 0):.4f}",
                'lr': f"{lr:.2e}",
            })
        
        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            # val_metrics = evaluate(model, val_loader, device)
            # print(f"\nStep {step}: Val Loss: {val_metrics['val_loss']:.4f}, PPL: {val_metrics['val_ppl']:.2f}")
            # if val_metrics['val_ppl'] < best_val_ppl:
            #     best_val_ppl = val_metrics['val_ppl']
            #     save_checkpoint(model, optimizer, scheduler, step, config, "best")
            pass
        
        # Save checkpoint
        if step % config.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, config, f"step_{step}")
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    print("Training complete!")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, config, "final")
    
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: TrainingConfig,
    name: str,
):
    """Save model checkpoint."""
    path = os.path.join(config.output_dir, f"{name}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step']


# Quick test training run
def test_training():
    """Run a quick training test."""
    config = TrainingConfig(
        model_config=NSTPConfig(
            vocab_size=1000,
            d_model=256,
            num_layers=4,
            num_heads=4,
            hsa_dim=4096,
            num_experts=4,
            top_k=2,
            d_ff=1024,
            router_tt_ranks=[1, 8, 8, 1],
            expert_tt_ranks=[1, 8, 8, 8, 1],
            embedding_tt_ranks=[1, 8, 8, 1],
            use_tt_embedding=True,
        ),
        seq_len=256,
        batch_size=4,
        max_steps=100,
        log_interval=10,
        eval_interval=50,
        save_interval=50,
        learning_rate=1e-3,
        warmup_steps=10,
        use_amp=False,
        output_dir="test_checkpoints",
    )
    
    model = train_nstp(config)
    print("Test training completed!")
    return model


if __name__ == "__main__":
    test_training()