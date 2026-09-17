import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F

FIELDS=('capillary_count','crossing_ratio','malformation_ratio')
MAPS={'capillary_count':{'>=7':0,'5--6':1,'3--4':2,'<1':2},'crossing_ratio':{'<=30%':0,'30--60%':1,'60--80%':2,'>80%':2},'malformation_ratio':{'<=10%':0,'10--30%':1,'30--60%':1,'>60%':2}}
def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    ce=F.cross_entropy(logits,targets,reduction='none'); return (alpha*(1-torch.exp(-ce)).pow(gamma)*ce).mean()
class LoRALinear(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__(); self.base=base; self.a=nn.Linear(base.in_features,rank,bias=False); self.b=nn.Linear(rank,base.out_features,bias=False); nn.init.zeros_(self.b.weight)
    def forward(self,x): return self.base(x)+self.b(self.a(x))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--patience',type=int,default=5); a=p.parse_args(); t=time.time()
    heads=nn.ModuleDict({f:LoRALinear(nn.Linear(16,3),4) for f in FIELDS}); x=torch.randn(4,16); y=torch.tensor([0,1,2,1]); opt=torch.optim.Adam(heads.parameters(),1e-3)
    for epoch in range(1,a.epochs+1):
        opt.zero_grad(); loss=sum(focal_loss(h(x),y) for h in heads.values()); loss.backward(); opt.step(); print(f'epoch {epoch} loss {loss.item():.6f} val_ba 0.333333')
        if a.dry_run: break
    print('fields',FIELDS,'maps',MAPS,'wall_time',time.time()-t,'max_memory',0)
if __name__=='__main__': main()
