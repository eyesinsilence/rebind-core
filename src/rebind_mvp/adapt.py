from .runtime import resolve_device,encoder_dimension,peak_memory
"""MQuAKE edit-group adaptation. Gold paths are controlled TRAINING evidence, never QA retrieval."""
import argparse, collections, json, math, os, pathlib, random, time
import numpy as np
import torch
from torch.nn import functional as F
from .audit import digest, write, append
from .source_reader import ReBindModule
from .pair_update import masked_softmax


class BatchedReBind(ReBindModule):
    """Same parameters and equations as ReBindModule, with an explicit batch axis."""
    def forward(self, candidate, roles, tokens, token_mask, links, valid, allowed):
        b,n,k,_=candidate.shape; s,t=tokens.shape[1:3]; d=self.project.out_features
        x=self.project(candidate); r=self.role(roles)[:,:,None,:].expand_as(x)
        a=x[:,:,None,:,None,:].expand(b,n,n,k,k,d); other=x[:,None,:,None,:,:].expand_as(a)
        ra=r[:,:,None,:,None,:].expand_as(a); rb=r[:,None,:,None,:,:].expand_as(a)
        z=self.initialize(torch.cat([a,other,ra,rb],-1)); memory=self.project(tokens)
        beliefs=None
        for _ in range(self.layers):
            context=(z*valid[:,None,:,None,:,None]).sum((2,4))/valid.sum((1,2))[:,None,None,None].clamp_min(1) if self.mode!='independent_binding' else torch.zeros_like(x)
            if beliefs is not None: context=context+beliefs.exp()[...,None]*x
            q=self.read_query(torch.cat([x,r,context],-1)).reshape(b,1,n*k,d).expand(b,s,n*k,d)
            read,_=self.source_attention(q.reshape(b*s,n*k,d),memory.reshape(b*s,t,d),memory.reshape(b*s,t,d),key_padding_mask=~token_mask.reshape(b*s,t),need_weights=False)
            local=torch.cat([q,read.reshape(b,s,n*k,d)],-1)
            scores=self.source_score(local).reshape(b,s,n,k).masked_fill(~valid[:,None],-torch.inf)
            value=self.source_value(local).reshape(b,s,n,k,d)
            special=(self.interpretation_special(local).reshape(b,s,n,k,2)*valid[:,None,:,:,None]).sum(3)/valid.sum(-1)[:,None,:,None].clamp_min(1)
            interpretations=torch.cat([scores,special],-1)
            weights=masked_softmax(scores,links&valid[:,None],dim=1)*(1-interpretations.softmax(-1)[...,-1])[...,None]
            values=(weights[...,None]*value).sum(1)
            linked=(values[:,:,None,:,None,:]+values[:,None,:,None,:,:])/2
            u=self.unary(torch.cat([x,values],-1)).squeeze(-1)
            if self.mode=='bp_rebind':
                p=self.pair(z).squeeze(-1); p=(p+p.permute(0,2,1,4,3))/2
                messages=x.new_zeros(b,n,n,k); uv=u.masked_fill(~valid,-torch.inf)
                edges=~torch.eye(n,dtype=torch.bool,device=x.device)[None,:,:,None]
                for _ in range(max(4,n)):
                    incoming=messages.sum(1)[:,:,None,:]-messages.transpose(1,2)
                    raw=torch.logsumexp((uv[:,:,None,:]+incoming)[...,None]+p,dim=3).masked_fill(~valid[:,None],-torch.inf)
                    messages=torch.where(edges&valid[:,None],raw-torch.logsumexp(raw,-1,keepdim=True),torch.zeros_like(raw))
                beliefs=(uv+messages.sum(1)).log_softmax(-1); z=z+linked
            elif self.mode!='independent_binding':
                left=self.triangle.left(z); right=self.triangle.right(z)
                maximum=z.new_full(z.shape[:-1],-torch.inf); denom=torch.zeros_like(maximum); numer=torch.zeros_like(z)
                ii=torch.arange(n,device=x.device)[:,None]; jj=torch.arange(n,device=x.device)[None,:]
                for bridge in range(n if k>1 else 0):
                    shape=(b,n,n,k,k,k-1,d)
                    l=left[:,:,bridge,:,1:][:,:,None,:,None,:,:].expand(shape)
                    rbridge=right[:,bridge,:,1:,:].permute(0,1,3,2,4)[:,None,:,None,:,:,:].expand(shape)
                    mask=valid[:,:,None,:,None,None]&valid[:,None,:,None,:,None]&valid[:,bridge,None,None,None,None,1:]
                    path=allowed[:,:,:,bridge]&(ii!=bridge)&(jj!=bridge)&(ii!=jj)
                    mask=mask&path[:,:,:,None,None,None]
                    g=self.triangle.gate(torch.cat([z.unsqueeze(-2).expand(shape),l,rbridge],-1)).squeeze(-1).masked_fill(~mask,-torch.inf)
                    m=torch.maximum(maximum,g.max(-1).values); safe=torch.where(torch.isfinite(m),m,torch.zeros_like(m))
                    previous=torch.exp(maximum-safe); weight=torch.exp(g-safe[...,None])
                    numer=numer*previous[...,None]+(l*rbridge*weight[...,None]).sum(-2)
                    denom=denom*previous+weight.sum(-1); maximum=m
                message=numer/denom.clamp_min(torch.finfo(z.dtype).tiny)[...,None]
                z=self.triangle.norm(z+self.triangle.update(torch.cat([z,message,linked],-1)))
                z=z*(valid[:,:,None,:,None]&valid[:,None,:,None,:])[...,None]
        u=self.unary(torch.cat([x,values],-1)).squeeze(-1).masked_fill(~valid,-torch.inf)
        p=self.pair(z).squeeze(-1); p=(p+p.permute(0,2,1,4,3))/2
        if self.mode=='independent_binding':p=p*0
        return dict(unary=u,pair=p,interpretation=interpretations,source=scores,frontier=self.frontier(torch.cat([x,values],-1)).squeeze(-1))


def prepare(c):
    import pyarrow.parquet as pq
    root=pathlib.Path(c['paths']['workdir']); out=root/'data/adapt';out.mkdir(exist_ok=True)
    raw=root/'data/raw/T-00000-of-00001.parquet'; rows=pq.read_table(raw).to_pylist()
    # Connect all cases whose paths contain the same edited subject/relation, including unchanged appearances.
    keys={tuple(t[:2]) for r in rows for t in r['edit_triples']}; parents=list(range(len(rows))); owners={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,r in enumerate(rows):
        for key in {tuple(t[:2]) for t in r['orig_triples']+r['new_triples']} & keys:
            if key in owners:parents[find(i)]=find(owners[key])
            owners[key]=i
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    ordered=sorted(groups.values(),key=lambda rr:digest([20260914,sorted(r['case_id'] for r in rr)]))
    n=len(ordered); split={}; group_id={}
    for i,rr in enumerate(ordered):
        name='train' if i<int(.70*n) else 'dev' if i<int(.85*n) else 'test'
        group=digest(sorted(r['case_id'] for r in rr))[:20]
        for r in rr:split[r['case_id']]=name;group_id[r['case_id']]=group
    pools=collections.defaultdict(set)
    for r in rows:
        if split[r['case_id']]!='train':continue
        for subject,relation,obj in r['orig_triples_labeled']+r['new_triples_labeled']:pools[relation].add(obj)
    all_entities=sorted({x for values in pools.values() for x in values})
    samples=[]
    for r in rows:
        old=r['orig_triples_labeled']; new=r['new_triples_labeled']
        assert len(old)==len(new)
        for variant,question in enumerate(r['questions']):
            rng=random.Random(int(digest([r['case_id'],variant])[:12],16))
            before=[old[0][0]]+[t[2] for t in old]; after=[new[0][0]]+[t[2] for t in new]
            descriptions=['starting entity mentioned in the question']+[f'entity filling the {t[1]} relation at hop {i+1}' for i,t in enumerate(old)]
            candidates=[]
            for a,b,rel in zip(before,after,['starting entity']+[t[1] for t in old]):
                options=list(dict.fromkeys([a,b])); neg=sorted(pools[rel]-set(options)) or [x for x in all_entities if x not in options]
                rng.shuffle(neg);extra=[x for x in all_entities if x not in options and x not in neg];rng.shuffle(extra)
                options+=(neg+extra)[:4-len(options)];rng.shuffle(options);candidates.append(['UNKNOWN']+options)
            catalog='Named entities, without asserting any relation: '+ '; '.join(sorted({x for v in candidates for x in v[1:]}))+'.'
            oldfacts=[f'{s}: {rel} is {obj}.' for s,rel,obj in old]
            changes=[i for i,(a,b) in enumerate(zip(old,new)) if a!=b]
            updates=[f'Updated fact: {new[i][0]}: {new[i][1]} is {new[i][2]}.' for i in changes]
            sources_before=[catalog]+oldfacts; sources_after=sources_before+updates
            # Random source order and candidate order prevent positional label shortcuts.
            order_before=list(range(len(sources_before)));order_after=list(range(len(sources_after)));rng.shuffle(order_before);rng.shuffle(order_after)
            labels_before=[v.index(x) for v,x in zip(candidates,before)];labels_after=[v.index(x) for v,x in zip(candidates,after)]
            def interpretations(source_order,is_after):
                ans=[]
                for idx in source_order:
                    targets=[len(candidates[0])+1]*len(candidates) # NOT_APPLICABLE
                    if 1<=idx<=len(old):
                        hop=idx-1
                        if not (is_after and hop in changes):targets[hop+1]=labels_before[hop+1]
                    elif idx>len(old):
                        hop=changes[idx-len(old)-1];targets[hop+1]=labels_after[hop+1]
                    ans.append(targets)
                return ans
            sample=dict(case_id=r['case_id'],qid=f'T_{r["case_id"]:06d}_q{variant}',split=split[r['case_id']],group=group_id[r['case_id']],question=question,candidates=candidates,roles=[question+' '+d for d in descriptions],candidate_texts=[[x+' '+d for x in v] for v,d in zip(candidates,descriptions)],before_docs=[sources_before[i] for i in order_before],after_docs=[sources_after[i] for i in order_after],before_target=labels_before,after_target=labels_after,before_interpretation=interpretations(order_before,False),after_interpretation=interpretations(order_after,True),revision=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b])
            samples.append(sample)
    manifest=dict(raw_hash=digest(raw),seed=20260914,split={s:sorted(r['case_id'] for r in rows if split[r['case_id']]==s) for s in ['train','dev','test']},groups={s:sorted({group_id[q] for q in split if split[q]==s}) for s in ['train','dev','test']},samples=len(samples),revision_labels=sum(len(s['revision']) for s in samples if s['split']=='train'),scope='Controlled old-to-updated fact transitions constructed from official MQuAKE-T paths. Gold path evidence and guaranteed candidate coverage used for supervised training/controlled diagnostics ONLY, never added to open QA. Test is a newly assigned training holdout after some prior frozen inference, not a previously untouched external benchmark.',candidate_catalog='Entities are available before the update without asserting relations. Distractor identities drawn only from training paths.',test_selection='No previous T correctness or answer predictions read when assigning splits.',source_scope='Stale fact applicability after an explicit update; does not label ambiguous old-source semantic reinterpretation.',config_hash=digest(c))
    assert all(set(manifest['groups'][a]).isdisjoint(manifest['groups'][b]) for a,b in [('train','dev'),('train','test'),('dev','test')])
    write(out/'split.json',manifest);write(out/'samples.json',samples)
    texts={('query',t) for s in samples for t in s['roles']+[x for v in s['candidate_texts'] for x in v]}
    texts|={('passage','MQuAKE fact\n'+t) for s in samples for t in s['before_docs']+s['after_docs']}
    write(out/'texts.json',[dict(key=digest([kind,t]),kind=kind,text=t) for kind,t in sorted(texts)])
    write(root/'reports/adapt_data_audit.json',manifest)
    print(json.dumps({**{s:len(manifest['split'][s]) for s in manifest['split']},'groups':{s:len(v) for s,v in manifest['groups'].items()},'revision_labels':manifest['revision_labels'],'texts':len(texts)}),flush=True)
    return manifest


def encode(c,shard):
    from .retrieval import E5
    root=pathlib.Path(c['paths']['workdir']); data=root/'data/adapt'; path=data/f'embeddings_{shard}.pt'
    identity=digest(dict(texts=digest(data/'texts.json'),model=c['models']['retriever_path'],format='e5_float32_pooled_fp16_tokens_v1'))
    if path.exists():assert torch.load(path,weights_only=True)['identity']==identity;return
    encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever'),batch=64); rows=[r for r in json.loads((data/'texts.json').read_text()) if int(r['key'][:8],16)%4==shard]; values={};start=time.time()
    for kind in ['query','passage']:
        rr=[r for r in rows if r['kind']==kind]
        for begin in range(0,len(rr),64):
            batch=rr[begin:begin+64];texts=[r['text'] for r in batch]
            if kind=='query':
                for r,v in zip(batch,encoder.encode(texts,kind)):values[r['key']]=torch.from_numpy(v)
            else:
                tokens,mask=encoder.encode(texts,kind,tokens=True)
                for r,t,m in zip(batch,tokens,mask):values[r['key']]=t[m].half().cpu()
            if begin%1024==0:print('encode',shard,kind,begin,len(rr),round(time.time()-start),flush=True)
    torch.save(dict(identity=identity,values=values),path)
    print('encoded',shard,len(values),round(time.time()-start),flush=True)


def batch_features(samples,embeddings,device):
    k=len(samples[0]['candidates'][0]); n=len(samples[0]['candidates']); b=len(samples)
    def lookup(kind,t):return embeddings[digest([kind,t])]
    candidate=torch.stack([torch.stack([torch.stack([lookup('query',t) for t in v]) for v in s['candidate_texts']]) for s in samples]).to(device)
    roles=torch.stack([torch.stack([lookup('query',t) for t in s['roles']]) for s in samples]).to(device)
    valid=torch.ones(b,n,k,dtype=torch.bool,device=device); allowed=torch.ones(b,n,n,n,dtype=torch.bool,device=device)
    features=[]
    for stage in ['before','after']:
        docs=[s[stage+'_docs'] for s in samples]; count=len(docs[0]); vectors=[[lookup('passage','MQuAKE fact\n'+t) for t in ds] for ds in docs];width=max(len(v) for vs in vectors for v in vs)
        tokens=torch.zeros(b,count,width,candidate.shape[-1]);mask=torch.zeros(b,count,width,dtype=torch.bool);links=torch.zeros(b,count,n,k,dtype=torch.bool)
        for bi,(ss,vs,ds) in enumerate(zip(samples,vectors,docs)):
            for si,(v,text) in enumerate(zip(vs,ds)):
                tokens[bi,si,:len(v)]=v;mask[bi,si,:len(v)]=True
                for i,cands in enumerate(ss['candidates']):
                    for j,surface in enumerate(cands):links[bi,si,i,j]=j==0 or surface in text
        features.append(dict(candidate=candidate,roles=roles,tokens=tokens.to(device),token_mask=mask.to(device),links=links.to(device),valid=valid,allowed=allowed))
    labels={stage:torch.tensor([s[stage+'_target'] for s in samples],device=device) for stage in ['before','after']}
    labels.update({stage+'_interpretation':torch.tensor([s[stage+'_interpretation'] for s in samples],device=device) for stage in ['before','after']})
    return features,labels


def loss_and_metrics(before,after,labels,revision_weight):
    b,n,k=before['unary'].shape; total=before['unary'].new_zeros(b); accuracy={}
    for stage,out in [('before',before),('after',after)]:
        target=labels[stage]
        unary=F.cross_entropy(out['unary'].flatten(0,1),target.flatten(),reduction='none').reshape(b,n).mean(1)
        pair_target=target[:,:,None]*k+target[:,None,:]
        pair=F.cross_entropy(out['pair'].flatten(3,4).reshape(-1,k*k),pair_target.flatten(),reduction='none').reshape(b,n,n)
        off=~torch.eye(n,dtype=torch.bool,device=target.device);pair=pair[:,off].mean(1)
        interp=F.cross_entropy(out['interpretation'].reshape(-1,k+2),labels[stage+'_interpretation'].flatten(),reduction='none').reshape(b,-1).mean(1)
        total+=.5*(unary+.25*pair+.25*interp)
        accuracy[stage+'_binding']=(out['unary'].argmax(-1)==target).float().mean(1)
    changed=labels['before']!=labels['after']; stable=~changed
    old=before['unary'].gather(-1,labels['after'][...,None]).squeeze(-1)-before['unary'].gather(-1,labels['before'][...,None]).squeeze(-1)
    new=after['unary'].gather(-1,labels['after'][...,None]).squeeze(-1)-after['unary'].gather(-1,labels['before'][...,None]).squeeze(-1)
    revision=(F.softplus(.2-(new-old.detach()))*changed).sum(1)/changed.sum(1).clamp_min(1)
    total+=revision_weight*revision
    prev=before['unary'].log_softmax(-1).gather(-1,labels['before'][...,None]).squeeze(-1).detach()
    curr=after['unary'].log_softmax(-1).gather(-1,labels['after'][...,None]).squeeze(-1)
    total+=.1*((curr-prev).square()*stable).sum(1)/stable.sum(1).clamp_min(1)
    success=(before['unary'].argmax(-1)==labels['before'])&(after['unary'].argmax(-1)==labels['after'])&changed
    accuracy.update(revision=success.sum(1)/changed.sum(1).clamp_min(1),loss=total)
    return total,accuracy


def train(c,method):
    root=pathlib.Path(c['paths']['workdir']);data=root/'data/adapt'; settings=c['adapt']; device=resolve_device(c,'train');torch.set_num_threads(8)
    samples=json.loads((data/'samples.json').read_text());embeddings={}
    for shard in range(4):embeddings.update(torch.load(data/f'embeddings_{shard}.pt',weights_only=True)['values'])
    identity=digest(dict(config=c,split=digest(data/'split.json'),samples=digest(data/'samples.json'),code=digest(pathlib.Path(__file__))))
    splits={name:[s for s in samples if s['split']==name] for name in ['train','dev','test']};batchsize=settings['batch_size']
    # Frozen features fit on each 49 GB card. Keep them resident to remove CPU collation from every optimizer step.
    buckets=collections.defaultdict(list);packed={};positions={}
    for sample in samples:buckets[(len(sample['candidates']),len(sample['before_docs']),len(sample['after_docs']))].append(sample)
    for key,rows in buckets.items():
        features,labels=batch_features(rows,embeddings,'cpu')
        packed[key]=([{k:v.to(device=device,dtype=torch.float16 if k=='tokens' else v.dtype) for k,v in f.items()} for f in features],{k:v.to(device) for k,v in labels.items()})
        positions.update({row['qid']:i for i,row in enumerate(rows)})
    del features,labels,embeddings
    print('resident_features_gib',(torch.cuda.memory_allocated(device) if device.startswith('cuda') else 0)/2**30,flush=True)
    def run_epoch(model,rows,opt=None,seed=0):
        model.train(opt is not None); groups=collections.defaultdict(list);rng=random.Random(seed)
        if opt:
            bygroup=collections.defaultdict(list)
            for row in rows:bygroup[row['group']].append(row)
            names=sorted(bygroup);rows=[rng.choice(bygroup[rng.choice(names)]) for _ in range(len(rows))]
        for row in rows:groups[(len(row['candidates']),len(row['before_docs']),len(row['after_docs']))].append(row)
        batches=[]
        for key,rr in groups.items():
            size=max(4,int(batchsize*(3/key[0])**4))
            batches.extend(rr[i:i+size] for i in range(0,len(rr),size))
        if opt:rng.shuffle(batches)
        recorded=collections.defaultdict(lambda:collections.defaultdict(list));gradients=[]
        for bi,batch in enumerate(batches):
            key=(len(batch[0]['candidates']),len(batch[0]['before_docs']),len(batch[0]['after_docs']))
            stored,targets=packed[key];indices=torch.tensor([positions[row['qid']] for row in batch],device=device)
            features=[{k:v.index_select(0,indices).float() if k=='tokens' else v.index_select(0,indices) for k,v in f.items()} for f in stored]
            labels={k:v.index_select(0,indices) for k,v in targets.items()}
            with torch.set_grad_enabled(opt is not None):
                before=model(**features[0]);after=model(**features[1]);loss,metrics=loss_and_metrics(before,after,labels,0 if method=='no_revision_loss' else .5)
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite adaptation loss')
                if opt:
                    opt.zero_grad(set_to_none=True);loss.mean().backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();gradients.append(float(grad))
            for key,value in metrics.items():
                for row,v in zip(batch,value.detach().cpu().tolist()):recorded[row['group']][key].append(v)
            if opt and bi%25==0:print('batch',method,seed,bi,len(batches),float(loss.mean().detach()),flush=True)
        result={key:float(np.mean([np.mean(values[key]) for values in recorded.values()])) for key in next(iter(recorded.values()))}
        if gradients:result['gradient_norm_max']=max(gradients)
        return result
    for seed in settings['seeds']:
        if settings.get('continue_pairs') and [method,seed] not in settings['continue_pairs']:continue
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        mode='rebind' if method=='no_revision_loss' else method
        model=BatchedReBind(input_dim=next(iter(packed.values()))[0][0]['candidate'].shape[-1],d=c['rebind']['hidden_dim'],layers=c['rebind']['layers'],mode=mode).to(device)
        initial=pathlib.Path(c['mquake']['checkpoint_root'])/method/str(seed)/'best.pt';model.load_state_dict(torch.load(initial,map_location=device,weights_only=True)['model'])
        directory=data/settings.get('checkpoint_directory','checkpoints')/method/str(seed);directory.mkdir(parents=True,exist_ok=True)
        if settings.get('resume_parent') and not (directory/'last.pt').exists():
            parent=data/settings['resume_parent']/method/str(seed)
            for name in ['best.pt','last.pt']:
                inherited=torch.load(parent/name,weights_only=True,map_location='cpu')
                assert inherited['identity']==settings['parent_identity']
                inherited.update(identity=identity,parent_identity=settings['parent_identity'],parent_checkpoint_hash=digest(parent/name))
                torch.save(inherited,directory/name)
        opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2,min_lr=1e-6)
        start_epoch=0;best=float('inf');bad=0
        if (directory/'complete.json').exists():
            assert json.loads((directory/'complete.json').read_text())['identity']==identity;continue
        if (directory/'last.pt').exists():
            saved=torch.load(directory/'last.pt',weights_only=True,map_location=device);assert saved['identity']==identity
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);start_epoch=saved['epoch']+1;best=saved['best'];bad=saved['bad']
        else:
            baseline=run_epoch(model,splits['dev']);append(root/'reports/adapt_learning_curves.jsonl',dict(method=method,seed=seed,epoch=-1,dev=baseline,scope='unadapted checkpoint on controlled MQuAKE dev',identity=identity))
            best=baseline['loss'];torch.save(dict(model=model.state_dict(),identity=identity,epoch=-1,initial_hash=digest(initial)),directory/'best.pt')
        for epoch in range(start_epoch,settings['max_epochs']):
            started=time.time();training=run_epoch(model,splits['train'],opt,seed+1000*epoch);dev=run_epoch(model,splits['dev']);scheduler.step(dev['loss'])
            improved=dev['loss']<best-settings['min_delta'];bad=0 if improved else bad+1
            if improved:best=dev['loss']
            saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,bad=bad,identity=identity,initial_hash=digest(initial),method=method,seed=seed)
            torch.save(saved,directory/'last.pt')
            if improved:torch.save(saved,directory/'best.pt')
            row=dict(method=method,seed=seed,epoch=epoch,train=training,dev=dev,seconds=time.time()-started,learning_rate=opt.param_groups[0]['lr'],best=best,bad=bad,identity=identity)
            append(root/'reports/adapt_learning_curves.jsonl',row);print(json.dumps(row),flush=True)
            if bad>=settings['patience']:break
        # Selection is complete before controlled test labels are scored. Open QA is separate.
        saved=torch.load(directory/'best.pt',weights_only=True,map_location=device);model.load_state_dict(saved['model']);test=run_epoch(model,splits['test'])
        result=dict(method=method,seed=seed,identity=identity,best_epoch=saved['epoch'],test_controlled=test,checkpoint=str(directory/'best.pt'),scope='Gold-path controlled diagnostic, not open QA',initial_hash=digest(initial))
        write(directory/'complete.json',result);print(json.dumps(result),flush=True)


def main():
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','encode','train']);parser.add_argument('--config',default='configs/adapt.yaml');parser.add_argument('--shard',type=int);parser.add_argument('--method');args=parser.parse_args()
    c=yaml.safe_load(pathlib.Path(args.config).read_text());root=pathlib.Path(c['paths']['workdir']);start=time.time();status='ok'
    try:
        if args.phase=='prepare':prepare(c)
        elif args.phase=='encode':encode(c,args.shard)
        else:train(c,args.method)
    except BaseException:
        status='failed';raise
    finally:append(root/'runs/adapt/commands.jsonl',dict(argv=__import__('sys').argv,start=start,seconds=time.time()-start,status=status,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),code_hash=digest(pathlib.Path(__file__))))

if __name__=='__main__':main()
