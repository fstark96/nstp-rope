"""Quick scaled training on WikiText-2 (SEQ=512, batch=2, grad_accum=32)"""
import sys, time, math, os, numpy as np, torch, torch.nn as nn
class FakeProfile:
    def run(*a, **k): pass
    def runctx(*a, **k): pass
sys.modules['profile'] = FakeProfile()
import torch.nn.functional as F
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_core.hsa import HSADenoiser
from nstp_core.moe import TTCERMoE

DEVICE = torch.device('cuda')
SEQ, VS, DM, NL, NH = 512, 50257, 320, 3, 4
HSA_DIM, NE, TK, DFF = 2048, 4, 2, 768
RTR, ETR, DROPOUT = [1,4,4,1], [1,4,4,4,1], 0.1
BATCH, GRAD_ACCUM = 2, 32  # effective = 2×512×32 = 32768 tokens
MAX_STEPS, EVAL_EVERY = 30000, 500
LR, CLIP = 3e-4, 1.0
CKPT = 'C:/Users/user/AppData/Local/Temp/nstp-v2/models_scaled/wiki2_seq512.pt'
os.makedirs(os.path.dirname(CKPT), exist_ok=True)

# Load WikiText-2 data
train_toks = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_train_tokens.npy')
val_toks   = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_validation_tokens.npy')
test_toks  = np.load('C:/Users/user/AppData/Local/Temp/nstp-v2/data/wikitext2_test_tokens.npy')

class DS:
    def __init__(self, t, seq):
        t = torch.tensor(t, dtype=torch.long); n = max(0, (len(t)-1)//seq)
        self.xs = torch.stack([t[i*seq:i*seq+seq] for i in range(n)])
        self.ys = torch.stack([t[i*seq+1:i*seq+seq+1] for i in range(n)])
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

train_ds = DS(train_toks, SEQ); val_ds = DS(val_toks, SEQ)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
val_ld   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH)

print(f"Train={len(train_ds)} Val={len(val_ds)} batches/step={GRAD_ACCUM}")

# Model (exact same as train_final.py)
class VH:
    @staticmethod
    def bind(h, pos, hd):
        freq = torch.fft.rfft(h, dim=-1); n = freq.shape[-1]; f = torch.arange(n, device=h.device, dtype=h.dtype)
        return torch.fft.irfft(freq * torch.complex(torch.cos(2*math.pi*f*pos.float().unsqueeze(-1)/hd), torch.sin(2*math.pi*f*pos.float().unsqueeze(-1)/hd)), n=hd, dim=-1)
    @staticmethod
    def unbind(M, pos, hd):
        B, S = pos.shape; M_exp = M.unsqueeze(1).expand(-1, S, -1); freq = torch.fft.rfft(M_exp, dim=-1)
        n = freq.shape[-1]; f = torch.arange(n, device=M.device, dtype=M.dtype)
        return torch.fft.irfft(freq * torch.complex(torch.cos(2*math.pi*f*(-pos.float()).unsqueeze(-1)/hd), torch.sin(2*math.pi*f*(-pos.float()).unsqueeze(-1)/hd)), n=hd, dim=-1)

class HDAAttn(nn.Module):
    def __init__(self, dm, hd, nh, drop=0.1):
        super().__init__()
        self.nh=nh; self.hdim=hd//nh
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(dm,self.hdim,bias=True),nn.LayerNorm(self.hdim)) for _ in range(nh)])
        self.denoisers=nn.ModuleList([HSADenoiser(self.hdim,3,False) for _ in range(nh)])
        self.out_proj=nn.Linear(hd,dm); self.drop=nn.Dropout(drop)
        for e in self.encoders: nn.init.xavier_uniform_(e[0].weight,gain=0.5); nn.init.zeros_(e[0].bias)
    def forward(self,x,pos):
        heads=[]
        for h in range(self.nh):
            h_enc=F.normalize(self.encoders[h](x),p=2,dim=-1)
            M=VH.bind(h_enc,pos,self.hdim).mean(dim=1)
            heads.append(self.denoisers[h](VH.unbind(M,pos,self.hdim)))
        return self.drop(self.out_proj(torch.cat(heads,dim=-1)))

class Block(nn.Module):
    def __init__(self,dm,hd,nh,ne,tk,dff,rtr,etr,drop=0.1):
        super().__init__()
        self.attn=HDAAttn(dm,hd,nh,drop)
        self.moe=TTCERMoE(d_model=dm,d_ff=dff,num_experts=ne,top_k=tk,router_tt_ranks=rtr,expert_tt_ranks=etr,activation='gelu',dropout=drop,router_aux_loss_coef=0.01)
        self.norm1=nn.LayerNorm(dm); self.norm2=nn.LayerNorm(dm); self.drop=nn.Dropout(drop)
    def forward(self,x,pos):
        r=x; x=self.norm1(x); a=self.attn(x,pos); x=r+self.drop(a)
        r=x; x=self.norm2(x); m,_=self.moe(x); return r+m

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VS,DM)
        self.blocks=nn.ModuleList([Block(DM,HSA_DIM,NH,NE,TK,DFF,RTR,ETR,DROPOUT) for _ in range(NL)])
        self.norm=nn.LayerNorm(DM); self.head=nn.Linear(DM,VS,bias=False)
        self.drop=nn.Dropout(DROPOUT)
        def _init(m):
            if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,0.02)
            elif isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        self.apply(_init)
    def forward(self,ids,pos=None):
        B,S=ids.shape; dev=ids.device
        if pos is None: pos=torch.arange(S,device=dev).unsqueeze(0).expand(B,-1)
        x=self.drop(self.embed(ids))
        for b in self.blocks: x=b(x,pos)
        return self.head(self.norm(x))

model=Model().to(DEVICE)
p=sum(p.numel() for p in model.parameters()); print(f"Params: {p:,} ({p/1e6:.1f}M)")

opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),weight_decay=0.1)
crit=nn.CrossEntropyLoss(reduction='mean')
scaler=torch.amp.GradScaler('cuda',enabled=True)
best=1e9; step=0; start=time.time(); t=0

print(f"{'Step':>6} {'TrCE':>8} {'ValPPL':>8} {'t/s':>6}")
print("-"*35)

while step<MAX_STEPS:
    for xb,yb in train_ld:
        if step>=MAX_STEPS: break
        xb,yb=xb.to(DEVICE),yb.to(DEVICE)
        with torch.amp.autocast('cuda',enabled=True):
            loss=crit(model(xb).view(-1,VS),yb.view(-1))
        scaler.scale(loss/GRAD_ACCUM).backward()
        if (step+1)%GRAD_ACCUM==0:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        step+=1
        if step%EVAL_EVERY==0:
            model.eval()
            tl,tt=0.0,0
            with torch.no_grad():
                for vx,vy in val_ld:
                    vx,vy=vx.to(DEVICE),vy.to(DEVICE)
                    tl+=crit(model(vx).view(-1,VS),vy.view(-1)).item()*vx.numel(); tt+=vx.numel()
            ppl=math.exp(tl/tt); t=time.time()-start if step==EVAL_EVERY else t
            note="*BEST*" if ppl<best else ""
            if ppl<best: best=ppl; torch.save({'model':model.state_dict(),'ppl':ppl,'step':step},CKPT)
            print(f"{step:>6} {loss.item():>8.4f} {ppl:>8.2f} {t:>5.1f}s {note}")
            model.train()
    # reshuffle

# Test
model.load_state_dict(torch.load(CKPT,map_location=DEVICE,weights_only=True)['model'])
model.eval(); tl,tt=0.0,0
with torch.no_grad():
    for x,y in torch.utils.data.DataLoader(DS(test_toks,SEQ),batch_size=BATCH):
        x,y=x.to(DEVICE),y.to(DEVICE); tl+=crit(model(x).view(-1,VS),y.view(-1)).item()*x.numel(); tt+=x.numel()
print(f"\nRESULT: Val={best:.2f}  Test={math.exp(tl/tt):.2f}")
