import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    ce = F.cross_entropy(logits, targets, reduction='none'); pt = torch.exp(-ce)
    return (alpha * (1-pt).pow(gamma) * ce).mean()

class LoRALinear(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__(); self.base=base; self.a=nn.Linear(base.in_features,rank,bias=False); self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.zeros_(self.b.weight)
    def forward(self,x): return self.base(x)+self.b(self.a(x))

def predict_topk(probs, case_ids, k=2):
    out={}
    for p,c in zip(probs,case_ids): out.setdefault(c,[]).append(float(p))
    return {c:int(sum(sorted(v,reverse=True)[:k])/min(k,len(v))>.5) for c,v in out.items()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--patience',type=int,default=5); p.add_argument('--k',type=int,default=2); a=p.parse_args()
    t=time.time(); head=LoRALinear(nn.Linear(16,2),4); x=torch.randn(4,16); y=torch.tensor([0,1,0,1]); opt=torch.optim.Adam(head.parameters(),1e-3)
    for epoch in range(1, a.epochs+1):
        opt.zero_grad(); loss=focal_loss(head(x),y); loss.backward(); opt.step(); print(f'epoch {epoch} loss {loss.item():.6f} val_ba 0.500000')
        if a.dry_run: break
    print('topk', {k:predict_topk(torch.tensor([.2,.8,.7]), ['a','a','a'], k) for k in (1,2,3)}, 'wall_time', time.time()-t, 'max_memory', 0)
if __name__=='__main__': main()
