"""
Download ~100M tokens of FineWeb-Edu, tokenize with GPT-2 BPE, save as .npy
Run with hermes venv Python.
"""
import sys
sys.path = [p for p in sys.path if 'hermes' not in p.lower() or 'venv' in p]

import os, time, numpy as np
from huggingface_hub import hf_hub_download
from transformers import GPT2TokenizerFast

DATA_DIR = 'C:/Users/user/AppData/Local/Temp/nstp-v2/data'
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

TARGET_TOKENS = 100_000_000  # 100M tokens
BATCH_SIZE = 1000  # texts per batch

print(f"Downloading FineWeb-Edu sample-10BT (streaming)...")
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', 
                  streaming=True, trust_remote_code=True)

all_tokens = []
total_tokens = 0
start = time.time()

for i, text in enumerate(ds):
    if total_tokens >= TARGET_TOKENS:
        break
    tokens = tokenizer.encode(text['text'], add_special_tokens=False)
    all_tokens.extend(tokens)
    total_tokens += len(tokens)
    
    if (i+1) % 10000 == 0:
        elapsed = time.time() - start
        rate = total_tokens / elapsed
        print(f"  texts={i+1:,}  tokens={total_tokens:,}/{TARGET_TOKENS:,}  "
              f"rate={rate/1e6:.1f}M tok/s  time={elapsed:.0f}s")

total_tokens = len(all_tokens)
print(f"\nTotal tokens: {total_tokens:,}")

# Convert to numpy array
all_tokens = np.array(all_tokens, dtype=np.int32)

# Split: 80% train, 10% val, 10% test
np.random.seed(42)
np.random.shuffle(all_tokens)
n = len(all_tokens)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_toks = all_tokens[:train_end]
val_toks = all_tokens[train_end:val_end]
test_toks = all_tokens[val_end:]

# Save
np.save(f'{DATA_DIR}/fineweb_train_tokens.npy', train_toks)
np.save(f'{DATA_DIR}/fineweb_val_tokens.npy', val_toks)
np.save(f'{DATA_DIR}/fineweb_test_tokens.npy', test_toks)

print(f"\nSaved to {DATA_DIR}/")
print(f"  Train: {len(train_toks):,} tokens")
print(f"  Val:   {len(val_toks):,} tokens")
print(f"  Test:  {len(test_toks):,} tokens")
print(f"  Total: {len(train_toks)+len(val_toks)+len(test_toks):,} tokens")
