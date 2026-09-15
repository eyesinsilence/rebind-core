import torch
from torch.nn import functional as F

def set_loss(logits,allowed,valid,mask):
    allowed=allowed&valid
    informative=mask&allowed.any(-1)&((valid&~allowed).any(-1))
    if not bool(informative.any()): return logits[torch.isfinite(logits)].sum()*0
    x=logits[informative].masked_fill(~valid[informative],-torch.inf)
    g=allowed[informative]
    return (torch.logsumexp(x,-1)-torch.logsumexp(x.masked_fill(~g,-torch.inf),-1)).mean()

def transition_loss(before,after,labels,revision_weight=.5,stable_weight=.1,frontier_weight=.2):
    terms=[]
    for output,key in [(before,'before'),(after,'after')]:
        for field in ['unary','pair','source','interpretation','frontier']:
            if field not in labels.get(key,{}): continue
            target=labels[key][field]
            x=output[field]; v=target['valid']; a=target['allowed']; mask=target['mask']
            if field=='pair': x=x.flatten(-2); v=v.flatten(-2); a=a.flatten(-2)
            value=set_loss(x,a,v,mask)
            terms.append(value*(frontier_weight if field=='frontier' else 1))
    if labels.get('revision'):
        losses=[]
        for i,a,b in labels['revision']:
            old=(before['unary'][i,b]-before['unary'][i,a]).detach()
            new=after['unary'][i,b]-after['unary'][i,a]
            losses.append(F.softplus(.2-(new-old)))
        terms.append(revision_weight*torch.stack(losses).mean())
    if labels.get('stable'):
        values=[]
        for i,a in labels['stable']:
            values.append((after['unary'][i].log_softmax(-1)[a]-before['unary'][i].log_softmax(-1)[a].detach()).square())
        terms.append(stable_weight*torch.stack(values).mean())
    return sum(terms,after['unary'][torch.isfinite(after['unary'])].sum()*0)
