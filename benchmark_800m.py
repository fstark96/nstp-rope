"""Benchmark NSTP v1 trained on 800M tokens."""
import sys, os, json, math, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from train_final import NSTPModel, DEVICE
from transformers import GPT2TokenizerFast

CONFIG = dict(vocab_size=50257, d_model=320, num_layers=3, num_heads=4,
              hsa_dim=2048, num_experts=4, top_k=2, d_ff=768,
              router_tt_ranks=[1,4,4,1], expert_tt_ranks=[1,4,4,4,1], dropout=0.1)

print("Loading 800M token model...")
model = NSTPModel(**CONFIG).to(DEVICE)
ckpt = torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models/nstp_800m.pt',
                   map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded. Val PPL: {ckpt['val_ppl']:.2f}")

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def score_ending(ctx, ending):
    full_text = ctx + " " + ending
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(ids) < 2: return float('-inf')
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 50257), y.view(-1), reduction='sum')
    return -loss.item()

# HellaSwag
print("\n" + "="*60)
print("BENCHMARK: HellaSwag (1000 examples)")
print("="*60)

HELLASWAG_FILE = "C:/Users/user/AppData/Local/Temp/nstp-v2/data/hellaswag_val.jsonl"
correct = 0; total = 0; start = time.time()
with open(HELLASWAG_FILE, 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000: break
        data = json.loads(line)
        scores = [score_ending(data['ctx'], e) for e in data['endings']]
        if np.argmax(scores) == data['label']: correct += 1
        total += 1
        if (i+1) % 200 == 0:
            print(f"  {i+1}/1000: {correct}/{total} ({100*correct/total:.1f}%)")

hs_acc = 100*correct/total
print(f"\nHellaSwag: {hs_acc:.2f}% ({correct}/{total}) in {time.time()-start:.0f}s")
print(f"Previous (2.4M tokens): 21.2%")
print(f"Reference: GPT-2 small ~52%, GPT-2 medium ~62%")

# PIQA
print("\n" + "="*60)
print("BENCHMARK: PIQA (1538 examples)")
print("="*60)

PIQA_FILE = "C:/Users/user/AppData/Local/Temp/nstp-v2/data/piqa_val.jsonl"
correct_piqa = 0; total_piqa = 0; start_piqa = time.time()
with open(PIQA_FILE, 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        scores = [score_ending(data['goal'], data[f'sol{i+1}']) for i in range(2)]
        if np.argmax(scores) == data['label']: correct_piqa += 1
        total_piqa += 1
        if (total_piqa) % 300 == 0:
            print(f"  {total_piqa}/1538: {correct_piqa}/{total_piqa} ({100*correct_piqa/total_piqa:.1f}%)")

piqa_acc = 100*correct_piqa/total_piqa
print(f"\nPIQA: {piqa_acc:.2f}% ({correct_piqa}/{total_piqa}) in {time.time()-start_piqa:.0f}s")
print(f"Reference: GPT-2 small ~60%, GPT-2 medium ~66%")

print(f"\n{'='*60}")
print("SUMMARY — NSTP v1 (800M tokens)")
print(f"{'='*60}")
print(f"HellaSwag: {hs_acc:.2f}% (was 21.2% with 2.4M tokens)")
print(f"PIQA:       {piqa_acc:.2f}%")
print(f"Val PPL:    {ckpt['val_ppl']:.2f}")