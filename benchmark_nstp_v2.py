"""
Custom benchmark script for NSTP v2 — no LM Eval Harness needed.
Evaluates on HellaSwag, MMLU, PIQA, GSM8K.
"""
import sys, os, json, math, time, numpy as np, torch
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_v2 import NSTPV2

DEVICE = torch.device('cuda')
print("Loading NSTP v2 model...")
model = NSTPV2(50257, 320, 3, 4, 2048, 4, 2, 768, 0.1).to(DEVICE)
ckpt = torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models_scaled/finetune_best.pt',
                   map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded: Val PPL={ckpt['ppl']:.2f}, Step={ckpt['step']}")

# Load GPT-2 tokenizer
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def compute_ppl(texts):
    """Compute perplexity on a list of texts."""
    total_loss, total_tokens = 0.0, 0
    crit = torch.nn.CrossEntropyLoss(reduction='none')
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        # Create input/target pairs
        x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
        y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(x)
            loss = crit(logits.view(-1, 50257), y.view(-1))
        total_loss += loss.sum().item()
        total_tokens += len(ids) - 1
    return math.exp(total_loss / total_tokens), total_loss / total_tokens

# ============================================================
# BENCHMARK 1: HellaSwag (subset — first 1000 examples)
# ============================================================
print("\n" + "="*60)
print("BENCHMARK: HellaSwag (subset)")
print("="*60)

# Download HellaSwag validation set
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
HELLASWAG_FILE = "C:/Users/user/AppData/Local/Temp/nstp-v2/data/hellaswag_val.jsonl"

if not os.path.exists(HELLASWAG_FILE):
    print("Downloading HellaSwag val...")
    import urllib.request
    urllib.request.urlretrieve(HELLASWAG_URL, HELLASWAG_FILE)
    print("Downloaded.")

correct = 0
total = 0
with open(HELLASWAG_FILE, 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        data = json.loads(line)
        ctx = data['ctx']
        label = data['label']
        endings = data['endings']
        
        # Score each ending
        scores = []
        for ending in endings:
            full_text = ctx + " " + ending
            ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(ids) < 2:
                scores.append(float('-inf'))
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
            y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, 50257), y.view(-1), reduction='sum')
            scores.append(-loss.item())  # Higher = better
        
        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/1000: {correct}/{total} ({100*correct/total:.1f}%)")

accuracy = 100 * correct / total
print(f"\nHellaSwag Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Reference: GPT-2 small ~52%, GPT-2 medium ~62%, GPT-2 large ~70%")

# ============================================================
# BENCHMARK 2: MMLU (5-shot subset)
# ============================================================
print("\n" + "="*60)
print("BENCHMARK: MMLU (subset)")
print("="*60)

# Download MMLU validation
MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_DIR = "C:/Users/user/AppData/Local/Temp/nstp-v2/data/mmlu"

if not os.path.exists(MMLU_DIR):
    print("Downloading MMLU...")
    os.makedirs(MMLU_DIR, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(MMLU_URL, os.path.join(MMLU_DIR, "data.tar"))
    import tarfile
    with tarfile.open(os.path.join(MMLU_DIR, "data.tar"), 'r') as tar:
        tar.extractall(MMLU_DIR)
    print("Downloaded and extracted.")

# Evaluate on a few MMLU subjects
subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology']
MMLU_VAL = os.path.join(MMLU_DIR, "data", "validation")

if os.path.exists(MMLU_VAL):
    correct = 0
    total = 0
    for subject in sorted(os.listdir(MMLU_VAL)):
        if not subject.endswith('.csv'):
            continue
        filepath = os.path.join(MMLU_VAL, subject)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                question = parts[0]
                answer = parts[5]  # Correct answer (A/B/C/D)
                
                # Score each choice
                scores = []
                for choice in ['A', 'B', 'C', 'D']:
                    prompt = f"Question: {question}\nAnswer: {choice}"
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    if len(ids) < 2:
                        scores.append(float('-inf'))
                        continue
                    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
                    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = model(x)
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, 50257), y.view(-1), reduction='sum')
                    scores.append(-loss.item())
                
                pred_idx = np.argmax(scores)
                pred_answer = chr(65 + pred_idx)  # A=65, B=66, etc.
                if pred_answer == answer:
                    correct += 1
                total += 1
                
                if total % 500 == 0:
                    print(f"  {total}: {correct}/{total} ({100*correct/total:.1f}%)")
        
        if total >= 5000:  # Limit to 5000 examples
            break
    
    accuracy = 100 * correct / total
    print(f"\nMMLU Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Reference: GPT-2 small ~25%, GPT-2 medium ~30%")
else:
    print(f"MMLU data not found at {MMLU_VAL}")

# ============================================================
# BENCHMARK 3: PIQA (first 1000 examples)
# ============================================================
print("\n" + "="*60)
print("BENCHMARK: PIQA (subset)")
print("="*60)

PIQA_URL = "https://raw.githubusercontent.com/ybansal/piqa/main/data/valid.jsonl"
PIQA_FILE = "C:/Users/user/AppData/Local/Temp/nstp-v2/data/piqa_val.jsonl"

if not os.path.exists(PIQA_FILE):
    print("Downloading PIQA val...")
    import urllib.request
    urllib.request.urlretrieve(PIQA_URL, PIQA_FILE)
    print("Downloaded.")

correct = 0
total = 0
with open(PIQA_FILE, 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        data = json.loads(line)
        question = data['question']
        sol1 = data['sol1']
        sol2 = data['sol2']
        label = data['label']  # 0 or 1
        
        scores = []
        for sol in [sol1, sol2]:
            full_text = question + " " + sol
            ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(ids) < 2:
                scores.append(float('-inf'))
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
            y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, 50257), y.view(-1), reduction='sum')
            scores.append(-loss.item())
        
        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/1000: {correct}/{total} ({100*correct/total:.1f}%)")

accuracy = 100 * correct / total
print(f"\nPIQA Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Reference: GPT-2 small ~72%, GPT-2 medium ~76%")

# ============================================================
# BENCHMARK 4: WikiText-2 PPL (reference)
# ============================================================
print("\n" + "="*60)
print("REFERENCE: WikiText-2 PPL")
print("="*60)
val_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

val_ds = DS(val_toks, 128)
test_ds = DS(test_toks, 128)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=8)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=8)

crit = nn.CrossEntropyLoss(reduction='mean')

for name, ld in [("Val", val_ld), ("Test", test_ld)]:
    tl, tt = 0.0, 0
    with torch.no_grad():
        for x, y in ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            lo, _ = model(x)
            tl += crit(lo.view(-1, 50257), y.view(-1)).item() * x.numel()
            tt += x.numel()
    ppl = math.exp(tl / tt)
    print(f"WikiText-2 {name} PPL: {ppl:.2f}")

print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
print(f"NSTP v2 (45.4M params, trained on FineWeb-Edu)")
print(f"  Val PPL: {ckpt['ppl']:.2f}")
print(f"  HellaSwag: {accuracy:.2f}% (if completed)")
print(f"  Reference: GPT-2 small ~52%, GPT-2 medium ~62%")
