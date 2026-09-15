"""Offline partial supervision and one equal-budget adaptation of deployed prefixes.

Private reference triples are read only by prepare(); never inserted into features.
"""
import collections, copy, json, pathlib, random, re, time
import torch
from .audit import digest, write, append


def relation_key(text):
    text=re.sub(r'[^a-z ]',' ',text.lower().replace('_',' '));text=' '.join(text.split())
    rules={
        'country of citizenship':r'citizen|nationality',
        'head of state':r'head of (the )?state',
        'head of government':r'head of (the )?government|leader of (the )?government|government leader|prime minister',
        'headquarters location':r'headquarter|main office',
        'country of origin':r'origin country|country of origin|originated in',
        'location of formation':r'location of formation|formed in|formation location',
        'place of death':r'place of death|died in|death occurred|life ended',
        'place of birth':r'place of birth|born in|birthplace',
        'employer':r'employer|employed by|works for|worked for',
        'educated at':r'educated at|studied at|attended|graduated from',
        'developer':r'developer|developed by',
        'manufacturer':r'manufacturer|manufactured by|produced by',
        'chief executive officer':r'chief executive officer|\bceo\b|top executive',
        'head coach':r'head coach', 'capital':r'\bcapital\b',
        'genre':r'\bgenre\b', 'sport':r'\bsport\b',
        'performer':r'performer|performed by',
        'original broadcaster':r'original broadcaster|originally broadcast',
        'notable work':r'notable work', 'work location':r'work location',
        'director / manager':r'has director|directed by|director of',
        'religion or worldview':r'religion|worldview',
        'child':r'\bchild\b|\bchildren\b', 'spouse':r'\bspouse\b|married to',
        'founded by':r'founded by', 'continent':r'\bcontinent\b',
        'creator':r'\bcreator\b', 'author':r'\bauthor\b|written by',
    }
    matches=[name for name,pattern in rules.items() if re.search(pattern,text)]
    return matches[0] if len(matches)==1 else None


def prefix_labels(trace,triples):
    """Semantic alias + directed reference edge + currently available input/value.

    Ambiguous generic 'leader', 'located in', 'created by' are deliberately masked.
    No UNKNOWN target is invented when the correct candidate is absent.
    """
    domain=trace['candidates'];variables=list(domain);n=len(variables);k=max(map(len,domain.values()));valid=torch.zeros(n,k,dtype=torch.bool);allowed=torch.zeros_like(valid);mask=torch.zeros(n,dtype=torch.bool)
    norm=lambda x:' '.join(re.findall(r'\w+',x.casefold()))
    known={};reachable={};audit=[];edges=[]
    for i,v in enumerate(trace['graph']['variables']):
        valid[i,:len(domain[v['var_id']])]=True
        if v.get('anchor'):
            valid[i,0]=False
            if norm(v['anchor'])==norm(triples[0][0]):
                known[v['var_id']]=triples[0][0];reachable[v['var_id']]=True
                for j,c in enumerate(domain[v['var_id']]):allowed[i,j]=norm(c['surface'])==norm(v['anchor'])
    for slot in trace['graph']['slots']:
        args=slot['ordered_arguments'];key=relation_key(slot['relation_text']);reason='unaligned_relation_or_direction';targets=[]
        if len(args)==2 and key and args[0] in known:
            targets=list({obj for subject,rel,obj in triples if norm(subject)==norm(known[args[0]]) and rel==key})
        if len(targets)==1:
            u,v=args;known[v]=targets[0];i=variables.index(v)
            for j,c in enumerate(domain[v]):
                inputs=c.get('input_values',{})
                admissible=c.get('origin_kind') in ['retrieved','parametric_hypothesis'] and inputs.get(u) is not None and norm(inputs[u])==norm(known[u]) and c.get('slot_id')==slot['slot_id']
                if reachable.get(u) and admissible and norm(c['surface'])==norm(targets[0]):allowed[i,j]=True
            mask[i]=allowed[i].any();reachable[v]=bool(mask[i]);reason='labeled' if mask[i] else 'correct_candidate_absent_or_upstream_unavailable'
            if mask[i]:edges.append((variables.index(u),i))
        audit.append(dict(slot=slot['slot_id'],relation=slot['relation_text'],canonical=key,status=reason))
    pair_allowed=torch.zeros(n,n,k,k,dtype=torch.bool);pair_mask=torch.zeros(n,n,dtype=torch.bool)
    for i,j in edges:
        pair_allowed[i,j]=allowed[i,:,None]&allowed[j,None,:];pair_mask[i,j]=pair_allowed[i,j].any()
    return dict(allowed=allowed,valid=valid,mask=mask,pair_allowed=pair_allowed,pair_mask=pair_mask),audit


def prepare(c,shard):
    import pyarrow.parquet as pq
    from .schema import QuestionGraph,CandidateRegistry
    from .transitions import FeatureBuilder
    from .retrieval import E5
    root=pathlib.Path(c['paths']['workdir']);directory=root/'data/final_alignment';directory.mkdir(exist_ok=True)
    raw={r['case_id']:r for r in pq.read_table(root/'data/raw/T-00000-of-00001.parquet').to_pylist()};splits=json.loads((root/'manifests/split_manifest.json').read_text());groups={i:g['group'] for g in splits['groups'] for i in g['cases']};train=set(json.loads((root/'manifests/alignment_prefix_lock.json').read_text())['train_cases']);dev=set(splits['dev_build']);assert not train&dev
    normalized=lambda text:' '.join(text.casefold().strip().rstrip('.').split())
    memory={normalized(d['text']):d['doc_id'] for d in json.loads((root/'data/memory/T.json').read_text())};triple_names={tuple(t):named for r in raw.values() for t,named in zip(r['new_triples'],r['new_triples_labeled'])};source_facts={}
    for r in raw.values():
        for request,triple in zip(r['requested_rewrite'],r['edit_triples']):
            doc_id=memory.get(normalized(request['prompt'].format(request['subject'])+' '+request['target_new_str']))
            if doc_id and tuple(triple) in triple_names:source_facts[doc_id]=triple_names[tuple(triple)]
    rows=[];seen=set();counts=collections.Counter();encoder=None;start=time.time()
    for split,stage,permitted in [('train','align_train_train_v4',train),('dev','dev_build_v4',dev)]:
        config=json.loads((root/'manifests'/(stage+'_lock.json')).read_text())['identity']['config'];builder=FeatureBuilder(None,config)
        for p in sorted((root/'runs/final'/stage).glob('T_*.json')):
            record=json.loads(p.read_text())
            if record['method'] not in ['json_fix','rebind_fix_s17'] or int(digest(record['qid'])[:8],16)%4!=shard:continue
            assert record['case_id'] in permitted
            if not record.get('compile_audit'):counts[split+'_frontend_failures']+=1;continue
            previous=None
            for t in record['trace']:
                key=digest([record['qid'],t['graph'],t['candidates'],t['proposal']]);path=directory/(key+'.pt')
                if key in seen:previous=str(path);continue
                seen.add(key);labels,audit=prefix_labels(t,raw[record['case_id']]['new_triples_labeled'])
                featurekey=digest(dict(feature_version='centered_window_v2',cache_context=dict(qid=record['qid'],split=stage),question=record['compile_audit']['question'],graph=t['graph'],candidates=t['candidates'],docs=t['proposal']['visible'],proposal=t['proposal'],config=config));feature=root/'data/features'/(featurekey+'.pt')
                if not feature.exists():
                    if encoder is None:encoder=E5(c['models']['retriever_path'],device='cuda:0')
                    builder.encoder=encoder;registry=CandidateRegistry(t['graph']['variables'])
                    registry.pool={v:{x['candidate_id']:x for x in xs} for v,xs in t['candidates'].items()};registry.active={v:[x['candidate_id'] for x in xs] for v,xs in t['candidates'].items()}
                    builder.build(record['compile_audit']['question'],QuestionGraph(**t['graph']),registry,t['proposal']['visible'],t['proposal'],dict(qid=record['qid'],split=stage));counts['feature_cache_misses']+=1
                else:counts['feature_cache_hits']+=1
                assert feature.exists();f=torch.load(feature,weights_only=True,map_location='cpu');assert torch.equal(labels['valid'],f['valid'])
                s=len(f['tokens']);n,k=labels['valid'].shape;ia=torch.zeros(s,n,k+2,dtype=torch.bool);im=torch.zeros(s,n,dtype=torch.bool)
                for si,di in enumerate(f['source_doc_index'].tolist()):
                    doc_id=t['proposal']['visible'][di]['doc_id']
                    for i,(v,xs) in enumerate(t['candidates'].items()):
                        for j,x in enumerate(xs):
                            fact=source_facts.get(doc_id);slot=next((slot for slot in t['graph']['slots'] if slot['slot_id']==x.get('slot_id')),None)
                            source_verified=fact and slot and relation_key(slot['relation_text'])==fact[1] and normalized(fact[0]) in [normalized(v) for v in x.get('input_values',{}).values()] and normalized(fact[2])==normalized(x['surface'])
                            if labels['mask'][i] and labels['allowed'][i,j] and x.get('origin_kind')=='retrieved' and doc_id in x.get('source_doc_ids',[]) and f['links'][si,i,j] and source_verified:ia[si,i,j]=True;im[si,i]=True
                labels.update(interpretation_allowed=ia,interpretation_mask=im)
                payload=dict(feature=str(feature),feature_hash=digest(feature),labels=labels,previous=previous,qid=record['qid'],case_id=record['case_id'],group=groups[record['case_id']],split=split,source_record=str(p),round=t['round'],graph=t['graph'],candidates=t['candidates'],top_assignment=(t['state'].get('assignments') or [{}])[0],visible_doc_ids=[d['doc_id'] for d in t['proposal']['visible']],audit=audit)
                torch.save(payload,path);rows.append(dict(path=str(path),sha256=digest(path),split=split,qid=record['qid'],group=groups[record['case_id']],unary=int(labels['mask'].sum()),pair=int(labels['pair_mask'].sum()),interpretation=int(im.sum())))
                counts[split+'_prefixes']+=1
                for name,value in [('unary',labels['mask']),('pair',labels['pair_mask']),('interpretation',im)]:counts[split+'_'+name+'_labels']+=int(value.sum())
                for a in audit:counts[split+'_'+a['status']]+=1
                previous=str(path)
    result=dict(shard=shard,rows=rows,counts=dict(counts),seconds=time.time()-start,labeler_hash=digest(pathlib.Path(__file__)),review='machine_rule_aligned; conservative semantic aliases, no human_valid',limitations=['Generic or ambiguous relation wording is masked.','Only current admissible candidate values can receive labels.','Interpretation labels are positive grounded sources only; no fabricated negative source labels.'])
    write(directory/f'prepare_{shard}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);return result


def collate(payloads,features,device):
    n=len(payloads[0]['labels']['valid']);assert all(len(p['labels']['valid'])==n for p in payloads)
    fs=[features[p['feature']] for p in payloads];b=len(fs);k=max(f['valid'].shape[1] for f in fs);s=max(len(f['tokens']) for f in fs);t=max(f['tokens'].shape[1] for f in fs);h=fs[0]['candidate'].shape[-1]
    f=dict(candidate=torch.zeros(b,n,k,h),roles=torch.zeros(b,n,h),tokens=torch.zeros(b,s,t,h),token_mask=torch.zeros(b,s,t,dtype=torch.bool),links=torch.zeros(b,s,n,k,dtype=torch.bool),valid=torch.zeros(b,n,k,dtype=torch.bool),allowed=torch.zeros(b,n,n,n,dtype=torch.bool))
    # Empty padding sources must have one zero token so attention cannot produce NaN.
    f['token_mask'][:,:,0]=True
    y=dict(allowed=torch.zeros(b,n,k,dtype=torch.bool),mask=torch.zeros(b,n,dtype=torch.bool),pair_allowed=torch.zeros(b,n,n,k,k,dtype=torch.bool),pair_mask=torch.zeros(b,n,n,dtype=torch.bool),interpretation_allowed=torch.zeros(b,s,n,k+2,dtype=torch.bool),interpretation_mask=torch.zeros(b,s,n,dtype=torch.bool))
    for j,(p,x) in enumerate(zip(payloads,fs)):
        kk=x['valid'].shape[1];ss,tt=x['tokens'].shape[:2];f['candidate'][j,:,:kk]=x['candidate'];f['roles'][j]=x['roles'];f['tokens'][j,:ss,:tt]=x['tokens'];f['token_mask'][j,:ss,:tt]=x['token_mask'];f['links'][j,:ss,:,:kk]=x['links'];f['valid'][j,:,:kk]=x['valid'];f['allowed'][j]=x['allowed'];l=p['labels']
        y['allowed'][j,:,:kk]=l['allowed'];y['mask'][j]=l['mask'];y['pair_allowed'][j,:,:,:kk,:kk]=l['pair_allowed'];y['pair_mask'][j]=l['pair_mask'];y['interpretation_allowed'][j,:ss,:,:kk]=l['interpretation_allowed'][...,:kk];y['interpretation_mask'][j,:ss]=l['interpretation_mask']
    return {k:v.to(device) for k,v in f.items()},{k:v.to(device) for k,v in y.items()}


def supervised_loss(out,f,y):
    from .train import set_loss
    unary=set_loss(out['unary'],y['allowed'],f['valid'],y['mask']);pv=f['valid'][:,:,None,:,None]&f['valid'][:,None,:,None,:]
    pair=set_loss(out['pair'].flatten(-2),y['pair_allowed'].flatten(-2),pv.flatten(-2),y['pair_mask'])
    iv=torch.cat([f['valid'][:,None].expand(-1,len(out['interpretation'][0]),-1,-1),torch.ones_like(out['interpretation'][...,-2:],dtype=torch.bool)],-1)
    interpretation=set_loss(out['interpretation'],y['interpretation_allowed'],iv,y['interpretation_mask'])
    return unary+.25*pair+.25*interpretation


def train(c,method,seed,epochs=5):
    from .adapt import BatchedReBind
    from .joint_decoder import decode,constraint_penalty
    from .schema import QuestionGraph
    root=pathlib.Path(c['paths']['workdir']);directory=root/'data/final_alignment';outdir=directory/method/str(seed);outdir.mkdir(parents=True,exist_ok=True);device='cuda:0';torch.set_num_threads(2);torch.manual_seed(seed)
    manifests=[json.loads((directory/f'prepare_{s}.json').read_text()) for s in range(4)];rows=[r for m in manifests for r in m['rows']];payloads={r['path']:torch.load(r['path'],weights_only=True,map_location='cpu') for r in rows};features={p['feature']:torch.load(p['feature'],weights_only=True,map_location='cpu') for p in payloads.values()};split={s:[p for p in payloads.values() if p['split']==s and p['labels']['mask'].any()] for s in ['train','dev']};assert split['train'] and split['dev'];assert {p['group'] for p in split['train']}.isdisjoint({p['group'] for p in split['dev']})
    selection=json.loads((pathlib.Path(c['diagnostics']['parent'])/'manifests/adapt_qa_lock.json').read_text())['identity']['all_trained_selections'][f'{method}_s{seed}'];model=BatchedReBind(d=c['rebind']['hidden_dim'],layers=c['rebind']['layers'],mode=method).to(device);model.load_state_dict(torch.load(selection['path'],weights_only=True,map_location=device)['model']);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    identity=dict(method=method,seed=seed,epochs=epochs,learning_rate=3e-5,batch_size=16,initial=selection,prefix_manifests=[digest(directory/f'prepare_{s}.json') for s in range(4)],code=digest(pathlib.Path(__file__)),batched_code=digest(root/'src/rebind_mvp/adapt.py'),supervision='unary + .25 pair + .25 positive source interpretation + .1 stable + .5 new-evidence correction opportunity',frontier_loss=False)
    lock=outdir/'lock.json'
    if lock.exists():raise ValueError('Training track already exists; preserve it, inspect completion before restarting')
    write(lock,identity);updates=0;best=(float('inf'),None);started=time.time();gradient_heads=set();epoch_results=[]
    for epoch in range(epochs+1):
        stats={}
        for name in (['dev'] if epoch==0 else ['train','dev']):
            training=name=='train';model.train(training);rr=split[name][:];rng=random.Random(seed+epoch)
            if training:rng.shuffle(rr)
            buckets=collections.defaultdict(list)
            for p in rr:buckets[len(p['labels']['valid'])].append(p)
            batches=[v[j:j+16] for v in buckets.values() for j in range(0,len(v),16)]
            if training:rng.shuffle(batches)
            losses=[];count=correct=0;stable_count=revision_count=0;joint_correct=joint_repairs=joint_harms=0
            for batch in batches:
                f,y=collate(batch,features,device)
                with torch.set_grad_enabled(training):
                    pred=model(**f);loss=supervised_loss(pred,f,y);stable=[];revision=[]
                    previous=[payloads.get(p['previous'],p) for p in batch];pf,py=collate(previous,features,device)
                    with torch.no_grad():old=model(**pf)
                    for bi,(p,prev) in enumerate(zip(batch,previous)):
                        if p is prev or p['graph']!=prev['graph']:continue
                        for i,v in enumerate(p['candidates']):
                            if not y['mask'][bi,i]:continue
                            now_ids={x['candidate_id']:j for j,x in enumerate(p['candidates'][v])};old_ids={x['candidate_id']:j for j,x in enumerate(prev['candidates'][v])}
                            current_indices=y['allowed'][bi,i].nonzero().flatten().tolist();old_indices=[old_ids[cid] for cid,j in now_ids.items() if j in current_indices and cid in old_ids and prev['labels']['allowed'][i,old_ids[cid]]]
                            if old_indices:
                                a=torch.logsumexp(pred['unary'][bi,i].log_softmax(-1)[current_indices],0);b=torch.logsumexp(old['unary'][bi,i].log_softmax(-1)[old_indices],0);stable.append((a-b).square())
                            wrong=prev['top_assignment'].get(v);wrong_indices=[j for j,x in enumerate(p['candidates'][v]) if x['surface']==wrong and j not in current_indices and x['surface']!='UNKNOWN']
                            new_evidence=any(x.get('origin_kind')=='retrieved' and set(x.get('source_doc_ids',[]))-set(prev['visible_doc_ids']) for j,x in enumerate(p['candidates'][v]) if j in current_indices)
                            if wrong_indices and new_evidence:
                                revision.append(torch.nn.functional.softplus(.2+torch.logsumexp(pred['unary'][bi,i,wrong_indices],0)-torch.logsumexp(pred['unary'][bi,i,current_indices],0)))
                    if stable:loss=loss+.1*torch.stack(stable).mean()
                    if revision:loss=loss+.5*torch.stack(revision).mean()
                    stable_count+=len(stable);revision_count+=len(revision)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite alignment loss')
                    if training:
                        opt.zero_grad(set_to_none=True);loss.backward()
                        for pname,param in model.named_parameters():
                            if param.grad is not None:
                                if not torch.isfinite(param.grad).all():raise ValueError('Nonfinite alignment gradient '+pname)
                                if torch.count_nonzero(param.grad):gradient_heads.add(pname.split('.')[0])
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();updates+=1
                losses.append(float(loss.detach()));chosen=pred['unary'].argmax(-1);count+=int(y['mask'].sum());correct+=int((y['allowed'].gather(-1,chosen[...,None]).squeeze(-1)&y['mask']).sum())
                if not training:
                    for bi,p in enumerate(batch):
                        g=QuestionGraph(**p['graph']);domain=p['candidates'];beam=decode(pred['unary'][bi],pred['pair'][bi],f['valid'][bi],beam=c['rebind']['joint_beam'],constraint=lambda a:constraint_penalty(g,domain,a))
                        for i,v in enumerate(domain):
                            if not y['mask'][bi,i]:continue
                            hit=bool(beam and y['allowed'][bi,i,beam[0][1][i]]);was=any(y['allowed'][bi,i,j] and x['surface']==p['top_assignment'].get(v) for j,x in enumerate(domain[v]));joint_correct+=hit;joint_repairs+=hit and not was;joint_harms+=was and not hit
            stats[name]=dict(loss=sum(losses)/len(losses),binding_correct=correct,binding_total=count,binding_accuracy=correct/count,stable_pairs=stable_count,new_evidence_correction_opportunities=revision_count)
            if not training:stats[name].update(joint_binding_correct=joint_correct,joint_binding_accuracy=joint_correct/count,joint_repairs_vs_collected=joint_repairs,joint_harms_vs_collected=joint_harms)
        checkpoint=outdir/f'epoch_{epoch}.pt';torch.save(dict(model=model.state_dict(),method=method,seed=seed,epoch=epoch,updates=updates,identity=identity,metrics=stats),checkpoint)
        if stats['dev']['loss']<best[0]:best=(stats['dev']['loss'],str(checkpoint))
        record=dict(epoch=epoch,updates=updates,stats=stats,seconds=time.time()-started,gradient_heads=sorted(gradient_heads),gpu_peak_bytes=torch.cuda.max_memory_allocated());epoch_results.append(record);append(outdir/'learning_curve.jsonl',record);print(json.dumps(record),flush=True)
    result=dict(method=method,seed=seed,updates=updates,epochs=epochs,best_masked_dev_checkpoint=best[1],selection_status='provisional; deployed dev QA and binding-harm comparison with epoch0 required before final lock',gradient_heads=sorted(gradient_heads),frontier_gradient='frontier' in gradient_heads,rows={s:len(v) for s,v in split.items()},results=epoch_results,identity=identity)
    assert not result['frontier_gradient'];write(outdir/'completion.json',result);return result
