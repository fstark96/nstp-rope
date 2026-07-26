"""Train NSTP v2 on FineWeb-Edu with all frontier model improvements"""
import sys, time, math, os, numpy as np, torch, torch.nn as nn
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_v2 import NSTPV2

DEVICE = torch.device('cuda')
SEQ, VS = 128, 50257
DM, NL, NH, HSA_DIM = 320, 3, 4, 2048
NE, TK, DFF = 4, 2, 768
BATCH = 32; MAX_STEPS = 30000; EVAL_EVERY = 1000
LR, CLIP = 3e-4, 1.0
CKPT = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models_scaled/nstp_v2_best.pt'
os.makedirs(os.path.dirname(CKPT), exist_ok=True)

train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/fineweb_val_tokens.npy')
print(f"Train={len(train_toks):,} Val={len(val_toks):,}")

class DS:
    def __init__(self,t,s):
        t=torch.tensor(t,dtype=torch.long); n=max(0,(len(t)-1)//s)
        self.xs=torch.stack([t[i*s:i*s+s] for i in range(n)])
        self.ys=torch.stack([t[i*s+1:i*s+s+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self,i): return self.xs[i],self.ys[i]

train_ds=DS(train_toks,SEQ); val_ds=DS(val_toks,SEQ)
train_ld=torch.utils.data.DataLoader(train_ds,batch_size=BATCH,shuffle=True,drop_last=True)
val_ld=torch.utils.data.DataLoader(val_ds,batch_size=BATCH)

# Load WK2 checkpoint for transfer learning
WK_CKPT = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models/final_best.pt'

model=NSTPV2(VS,DM,NL,NH,HSA_DIM,NE,TK,DFF,dropout=0.1,use_memory=True,memory_size=512).to(DEVICE)

# Try to load WK2 weights (partial transfer)
if os.path.exists(WK_CKPT):
    sd=torch.load(WK_CKPT,map_location=DEVICE,weights_only=True)
    missing,unexpected=model.load_state_dict(sd,strict=False)
    print(f"Loaded WK2: {len(sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
else:
    print("Fresh start (no WK2 checkpoint)")

p=sum(p.numel() for p in model.parameters()); print(f"Params: {p:,} ({p/1e6:.1f}M)")

opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
crit=nn.CrossEntropyLoss(reduction='mean')
scaler=torch.amp.GradScaler('cuda',enabled=True)

best=1e9; step=0; start=time.time()
print(f"{'Step':>6} {'TrCE':>8} {'ValPPL':>8} {'Time':>6}")
print("-"*35)

while step<MAX_STEPS:
    for xb,yb in train_ld:
        if step>=MAX_STEPS: break
        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
        with torch.amp.autocast('cuda',enabled=True):
            logits,aux_loss=model(xb)
            loss=crit(logits.view(-1,VS),yb.view(-1))+0.01*aux_loss
        scaler.scale(loss).backward()
        if (step+1)%16==0:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        step+=1
        if step%EVAL_EVERY==0:
            model.eval(); tl,tt=0.0,0
            with torch.no_grad():
                for vx,vy in val_ld:
                    vx,vy=vx.to(DEVICE),vy.to(DEVICE)
                    lo,_=model(vx)
                    tl+=crit(lo.view(-1,VS),vy.view(-1)).item()*vx.numel(); tt+=vx.numel()
            ppl=math.exp(tl/tt); el=time.time()-start
            note="*BEST*" if ppl<best else ""
            if ppl<best: best=ppl; torch.save({'model':model.state_dict(),'ppl':ppl,'step':step},CKPT)
            print(f"{step:>6} {loss.item():>8.4f} {ppl:>8.2f} {el:>5.0f}s {note}")
            model.train()

model.load_state_dict(torch.load(CKPT,map_location=DEVICE,weights_only=True)['model'])
model.eval(); tl,tt=0.0,0
with torch.no_grad():
    for x,y in torch.utils.data.DataLoader(DS(val_toks,SEQ),batch_size=BATCH):
        x,y=x.to(DEVICE),y.to(DEVICE)
        lo,_=model(x); tl+=crit(lo.view(-1,VS),y.view(-1)).item()*x.numel(); tt+=x.numel()
print(f"\nFINAL: Val={best:.2f}  Test={math.exp(tl/tt):.2f}  Steps={step}")
