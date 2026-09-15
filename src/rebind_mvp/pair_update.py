"""Candidate-matched gated triangle update, streamed over bridges and target rows."""
import torch
from torch import nn

def masked_softmax(x,mask,dim=-1):
    y=x.masked_fill(~mask,-torch.inf)
    y=torch.where(mask.any(dim,keepdim=True),y,torch.zeros_like(y))
    return y.softmax(dim)*mask.to(x.dtype)

class TriangleUpdate(nn.Module):
    def __init__(self,d,row_chunk=4):
        super().__init__(); self.row_chunk=row_chunk
        self.left=nn.Linear(d,d,bias=False); self.right=nn.Linear(d,d,bias=False)
        self.gate=nn.Linear(3*d,1); self.update=nn.Sequential(nn.Linear(3*d,2*d),nn.GELU(),nn.Linear(2*d,d)); self.norm=nn.LayerNorm(d)
    def message(self,z,valid,allowed,identity=False):
        n,_,k,_,d=z.shape
        if k==1: return z*0
        parts=[]; left=z if identity else self.left(z); right=z if identity else self.right(z)
        for begin in range(0,n,self.row_chunk):
            end=min(n,begin+self.row_chunk); target=z[begin:end]; shape=(*target.shape[:-1],k-1,d)
            maximum=z.new_full(target.shape[:-1],-torch.inf); denom=z.new_zeros(target.shape[:-1]); numer=torch.zeros_like(target)
            ii=torch.arange(begin,end,device=z.device)[:,None]; jj=torch.arange(n,device=z.device)[None,:]
            for bridge in range(n):
                # The candidate axis is shared, never independently reduced on the two edges.
                l=left[begin:end,bridge,:,1:][:,None,:,None,:,:].expand(shape)
                r=right[bridge,:,1:,:].permute(0,2,1,3)[None,:,None,:,:,:].expand(shape)
                mask=valid[begin:end,None,:,None,None]&valid[None,:,None,:,None]&valid[bridge,1:]
                path=allowed[begin:end,:,bridge]&(ii!=bridge)&(jj!=bridge)&(ii!=jj)
                mask=mask&path[:,:,None,None,None]
                g=z.new_zeros(shape[:-1]) if identity else self.gate(torch.cat([target.unsqueeze(-2).expand(shape),l,r],-1)).squeeze(-1)
                g=g.masked_fill(~mask,-torch.inf); m=torch.maximum(maximum,g.max(-1).values)
                safe=torch.where(torch.isfinite(m),m,torch.zeros_like(m)); previous=torch.exp(maximum-safe); weight=torch.exp(g-safe[...,None])
                numer=numer*previous[...,None]+((l*r)*weight[...,None]).sum(-2)
                denom=denom*previous+weight.sum(-1); maximum=m
            parts.append(numer/denom.clamp_min(torch.finfo(z.dtype).tiny)[...,None])
        return torch.cat(parts,0)
    def forward(self,z,valid,allowed,source):
        m=self.message(z,valid,allowed)
        out=self.norm(z+self.update(torch.cat([z,m,source],-1)))
        mask=valid[:,None,:,None]&valid[None,:,None,:]
        return out*mask[...,None]
