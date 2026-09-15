"""Log-space sum-product. Pair[i,j,a,b] uses source candidate a, target b."""
import torch

def belief_propagation(unary,pair,valid,edges,steps=4,damping=0.):
    n,k=unary.shape; messages=unary.new_zeros(n,n,k)
    u=unary.masked_fill(~valid,-torch.inf)
    for _ in range(steps):
        rows=[]
        for i in range(n):
            row=[]
            for j in range(n):
                if i==j or not bool(edges[i,j]): row.append(torch.zeros_like(u[j])); continue
                incoming=messages[:,i].sum(0)-messages[j,i]
                raw=torch.logsumexp((u[i]+incoming)[:,None]+pair[i,j],dim=0)
                raw=raw.masked_fill(~valid[j],-torch.inf)
                if not torch.isfinite(raw).any(): raise ValueError('Inconsistent BP hard constraints')
                normalized=raw-torch.logsumexp(raw,dim=0)
                # Probability damping; safe because both normalized distributions have identical domain.
                if damping:
                    old=messages[i,j].masked_fill(~valid[j],-torch.inf).log_softmax(-1)
                    normalized=torch.logaddexp(normalized+torch.log(raw.new_tensor(1-damping)),old+torch.log(raw.new_tensor(damping)))
                row.append(torch.where(valid[j],normalized,torch.zeros_like(normalized)))
            rows.append(torch.stack(row))
        messages=torch.stack(rows)
    logits=(u+messages.sum(0)).masked_fill(~valid,-torch.inf)
    return logits.log_softmax(-1),messages
