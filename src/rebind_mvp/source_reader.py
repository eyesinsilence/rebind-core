import torch
from torch import nn
from .pair_update import TriangleUpdate,masked_softmax
from .bp import belief_propagation

class ReBindModule(nn.Module):
    def __init__(self,input_dim=768,d=128,layers=4,mode='rebind'):
        super().__init__(); self.layers=layers; self.mode=mode
        self.project=nn.Linear(input_dim,d); self.role=nn.Linear(input_dim,d)
        self.initialize=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d))
        self.read_query=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,d))
        self.source_attention=nn.MultiheadAttention(d,4,batch_first=True,dropout=0.)
        self.interpretation_special=nn.Linear(2*d,2); self.source_score=nn.Linear(2*d,1); self.source_value=nn.Linear(2*d,d)
        self.triangle=TriangleUpdate(d); self.unary=nn.Linear(2*d,1); self.pair=nn.Linear(d,1); self.frontier=nn.Linear(2*d,1)
    def forward(self,candidate,roles,tokens,token_mask,links,valid,allowed,freeze=None,source_doc_index=None):
        # candidate[N,K,H], roles[N,H], source tokens[S,T,H], links[S,N,K].
        n,k,_=candidate.shape; x=self.project(candidate); r=self.role(roles)[:,None,:].expand_as(x)
        a=x[:,None,:,None,:].expand(n,n,k,k,-1); b=x[None,:,None,:,:].expand_as(a)
        ra=r[:,None,:,None,:].expand_as(a); rb=r[None,:,None,:,:].expand_as(a)
        z=self.initialize(torch.cat([a,b,ra,rb],-1)); memory=self.project(tokens)
        source_scores=[]; values=x.new_zeros(x.shape); beliefs=None
        for layer in range(self.layers):
            context=(z*valid[None,:,None,:,None]).sum((1,3))/valid.sum().clamp_min(1) if self.mode!='independent_binding' else torch.zeros_like(x)
            if beliefs is not None: context=context+beliefs.exp()[...,None]*x
            query=self.read_query(torch.cat([x,r,context],-1)).reshape(1,n*k,-1).expand(tokens.shape[0],-1,-1)
            read,_=self.source_attention(query,memory,memory,key_padding_mask=~token_mask,need_weights=False)
            local=torch.cat([query,read],-1)
            scores=self.source_score(local).reshape(-1,n,k)
            value=self.source_value(local).reshape(-1,n,k,x.shape[-1])
            if freeze is not None:
                oldmask,oldscore,oldvalue,oldspecial=freeze
                scores=torch.where(oldmask[:,None,None],oldscore,scores)
                value=torch.where(oldmask[:,None,None,None],oldvalue,value)
            special=(self.interpretation_special(local).reshape(-1,n,k,2)*valid[None,:,:,None]).sum(2)/valid.sum(-1)[None,:,None].clamp_min(1)
            if freeze is not None: special=torch.where(oldmask[:,None,None],oldspecial,special)
            scores=scores.masked_fill(~valid[None],-torch.inf)
            interpretations=torch.cat([scores,special],-1)
            # Source evidence aggregated once per deduplicated source, recomputed at every layer.
            applicability=1-interpretations.softmax(-1)[...,-1]
            weights=masked_softmax(scores,links.bool()&valid[None],dim=0)*applicability[...,None]
            values=(weights[...,None]*value).sum(0)
            source_scores.append(scores)
            linked=(values[:,None,:,None,:]+values[None,:,None,:,:])/2
            u=self.unary(torch.cat([x,values],-1)).squeeze(-1)
            if self.mode=='bp_rebind':
                potential=(self.pair(z).squeeze(-1)+self.pair(z).squeeze(-1).permute(1,0,3,2))/2
                edges=~torch.eye(n,dtype=torch.bool,device=x.device)
                beliefs,_=belief_propagation(u,potential,valid,edges,steps=max(4,n))
                z=z+linked
            elif self.mode!='independent_binding': z=self.triangle(z,valid,allowed,linked)
        u=self.unary(torch.cat([x,values],-1)).squeeze(-1).masked_fill(~valid,-torch.inf)
        p=self.pair(z).squeeze(-1); p=(p+p.permute(1,0,3,2))/2
        if self.mode=='independent_binding': p=p*0
        return dict(unary=u,pair=p,source=scores,interpretation=interpretations,source_special=special,source_layers=source_scores,source_values=value,frontier=self.frontier(torch.cat([x,values],-1)).squeeze(-1),z=z)
