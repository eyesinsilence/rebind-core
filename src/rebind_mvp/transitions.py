import json,pathlib,os
import torch
from .audit import digest,write
from .source_reader import ReBindModule
from .joint_decoder import decode,constraint_penalty
from .runtime import resolve_device,encoder_dimension,peak_memory

class FeatureBuilder:
    def __init__(self,encoder,config):
        self.encoder=encoder; self.root=pathlib.Path(config['paths']['data_root'])/'features'; self.root.mkdir(parents=True,exist_ok=True); self.config=config
    def build(self,question,graph,registry,docs,proposal,cache_context=None):
        key=digest(dict(feature_version='dimension_empty_safe_v3',cache_context=cache_context,question=question,graph=graph.__dict__,candidates=registry.snapshot(),docs=docs,proposal=proposal,config=self.config))
        file=self.root/(key+'.pt')
        if file.exists(): return torch.load(file,weights_only=True)
        domain=registry.snapshot(); vars=list(domain); n=len(vars); k=max(len(domain[v]) for v in vars)
        valid=torch.zeros(n,k,dtype=torch.bool)
        texts=[]; positions=[]
        for i,v in enumerate(vars):
            for j,c in enumerate(domain[v]): texts.append(c['surface']+' '+graph.variables[i]['description']); positions.append((i,j)); valid[i,j]=True
        vectors=self.encoder.encode(texts,'query')
        h=vectors.shape[-1];candidate=torch.zeros(n,k,h)
        for vec,(i,j) in zip(vectors,positions): candidate[i,j]=torch.from_numpy(vec)
        roles=torch.from_numpy(self.encoder.encode([question+' '+v['description']+' '+json.dumps([slot for slot in graph.slots if v['var_id'] in slot['ordered_arguments']],sort_keys=True) for v in graph.variables],'query'))
        tokens=[]; masks=[]; windows=[]; source_indices=[]
        # Source windows are centered on public, validated proposal quotes; never on gold spans.
        for di,doc in enumerate(docs):
            offsets=self.encoder.tokenizer(doc['text'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            title_length=len(self.encoder.tokenizer.encode('passage: '+doc['title']+'\n',add_special_tokens=False)); budget=max(16,512-title_length-8)
            quotes=[p for p in proposal.get('accepted',[]) if p['doc_id']==doc['doc_id']]
            centers=[max(0,p['char_start']-doc['offsets'][0]) for p in quotes] or [0]
            ranges=set()
            for center in centers:
                position=next((i for i,(a,b) in enumerate(offsets) if b>center),0); first=max(0,position-budget//2); last=min(len(offsets),first+budget)
                start=offsets[first][0] if offsets else 0; end=offsets[last-1][1] if offsets else 0; ranges.add((start,end))
            for start,end in sorted(ranges):
                window=dict(doc,text=doc['text'][start:end]); windows.append(window); source_indices.append(di)
                t,m=self.encoder.encode([doc['title']+'\n'+window['text']],'passage',tokens=True); tokens.append(t[0].cpu()); masks.append(m[0].cpu())
        if not tokens:
            # One numerically safe, unlinked padding source; -1 is never a document identity.
            tokens=[torch.zeros(1,h)];masks=[torch.ones(1,dtype=torch.bool)];source_indices=[-1]
        maxlen=max(len(x) for x in tokens); memory=torch.zeros(len(tokens),maxlen,h); tokenmask=torch.zeros(len(tokens),maxlen,dtype=torch.bool)
        for s,(t,m) in enumerate(zip(tokens,masks)):
            memory[s,:len(t)]=t;tokenmask[s,:len(m)]=m
        links=torch.zeros(len(tokens),n,k,dtype=torch.bool)
        for s,(t,m,doc) in enumerate(zip(tokens,masks,windows)):
            memory[s,:len(t)]=t; tokenmask[s,:len(m)]=m
            for i,v in enumerate(vars):
                links[s,i,0]=True
                for j,c in enumerate(domain[v][1:],1): links[s,i,j]=c['surface'] in doc['text'] or c['surface'] in doc['title']
        if self.config.get('final_plan'):
            for i,v in enumerate(vars):
                if graph.variables[i].get('anchor'):valid[i,0]=False
                for j,c in enumerate(domain[v][1:],1):
                    origins=set(c.get('source_doc_ids',[]))
                    for s,doc in enumerate(windows):links[s,i,j]=c.get('origin_kind')=='retrieved' and doc['doc_id'] in origins and links[s,i,j]
        result=dict(source_doc_index=torch.tensor(source_indices),candidate=candidate,roles=roles,tokens=memory,token_mask=tokenmask,links=links,valid=valid,allowed=torch.ones(n,n,n,dtype=torch.bool))
        torch.save(result,file); return result

class NeuralState:
    def __init__(self,encoder,config,checkpoint,mode='rebind',device=None,split='inference'):
        device=resolve_device(config,'scorer',device)
        self.features=FeatureBuilder(encoder,config); self.config=config; self.device=device; self.mode=mode; self.split=split
        ckpt=torch.load(checkpoint,weights_only=True,map_location=device)
        dimension=ckpt['model']['project.weight'].shape[1]
        if hasattr(encoder,'embedding_dim') and dimension!=encoder_dimension(encoder):raise ValueError('Checkpoint input dimension differs from encoder; use a matching checkpoint or retrain')
        self.model=ReBindModule(input_dim=dimension,d=config['rebind']['hidden_dim'],layers=config['rebind']['layers'],mode='rebind' if mode in ['frozen_old_read','no_revision_loss'] else mode).to(device)
        self.metadata={k:v for k,v in ckpt.items() if k not in ['model','optimizer']}; self.model.load_state_dict(ckpt['model']); self.model.eval(); self.previous={}; self.old_reads={}
    @torch.no_grad()
    def update(self,example,graph,registry,proposal,docs,round_index):
        f={k:v.to(self.device) for k,v in self.features.build(example.question,graph,registry,docs,proposal,dict(qid=example.qid,split=self.split)).items()}
        domain=registry.snapshot(); vars=list(domain); freeze=None
        if self.config.get('final_plan'):
            from .assignments import admissible
            policy=self.config['final_plan'].get('relation_policy','literal')
            for i,v in enumerate(vars):
                for j,candidate in enumerate(domain[v]):
                    f['valid'][i,j] &= admissible(candidate,policy)
        if self.mode=='frozen_old_read' and docs:
            # Freeze ALL old sources by first-exposure candidate IDs, including their special logits.
            # A newly proposed candidate gets zero old-source value; no new window bypasses the lesion.
            rows=[]; n,k=f['valid'].shape; d=self.config['rebind']['hidden_dim']
            for di,doc in enumerate(docs):
                if doc['doc_id'] in self.old_reads:
                    for saved in self.old_reads[doc['doc_id']]:
                        score=f['candidate'].new_full((n,k),-1e4); value=f['candidate'].new_zeros(n,k,d); link=torch.zeros(n,k,dtype=torch.bool,device=self.device)
                        for i,v in enumerate(vars):
                            for a,cand in enumerate(domain[v]):
                                if cand['candidate_id'] in saved['ids'][v]:
                                    b=saved['ids'][v].index(cand['candidate_id']); score[i,a]=saved['score'][i,b]; value[i,a]=saved['value'][i,b]; link[i,a]=saved['link'][i,b]
                        rows.append(dict(tokens=saved['tokens'],mask=saved['mask'],link=link,score=score,value=value,special=saved['special'],old=True,di=di))
                else:
                    for si in (f['source_doc_index']==di).nonzero().flatten().tolist():
                        rows.append(dict(tokens=f['tokens'][si],mask=f['token_mask'][si],link=f['links'][si],score=f['candidate'].new_zeros(n,k),value=f['candidate'].new_zeros(n,k,d),special=f['candidate'].new_zeros(n,2),old=False,di=di))
            width=max(len(row['tokens']) for row in rows); memory=f['tokens'].new_zeros(len(rows),width,f['tokens'].shape[-1]); mask=torch.zeros(len(rows),width,dtype=torch.bool,device=self.device)
            for si,row in enumerate(rows): memory[si,:len(row['tokens'])]=row['tokens']; mask[si,:len(row['mask'])]=row['mask']
            f.update(tokens=memory,token_mask=mask,links=torch.stack([r['link'] for r in rows]),source_doc_index=torch.tensor([r['di'] for r in rows],device=self.device))
            freeze=(torch.tensor([r['old'] for r in rows],device=self.device),torch.stack([r['score'] for r in rows]),torch.stack([r['value'] for r in rows]),torch.stack([r['special'] for r in rows]))
        activity=None
        if self.config.get('final_plan'):
            measured=[];original_message=self.model.triangle.message
            def observe(*args,**kwargs):
                message=original_message(*args,**kwargs)
                if not torch.isfinite(message).all():raise ValueError('Nonfinite triangle message')
                measured.append(int(torch.count_nonzero(message)));return message
            self.model.triangle.message=observe
            try:out=self.model(**f,freeze=freeze)
            finally:self.model.triangle.message=original_message
            if not torch.isfinite(out['unary'][f['valid']]).all() or not torch.isfinite(out['pair']).all():raise ValueError('Nonfinite operator output')
            counts=f['valid'].sum(-1).tolist();bridge_counts=f['valid'][:,1:].sum(-1).tolist();n=len(counts)
            legal=sum(counts[i]*counts[j]*bridge_counts[k] for i in range(n) for j in range(n) for k in range(n) if len({i,j,k})==3 and f['allowed'][i,j,k])
            activity=dict(variables=n,valid_candidates=counts,interpretation_graph_nodes=0,legal_message_paths=legal,nonzero_messages_by_layer=measured,frontier_score_used_for_query=False,candidate_tensor_shape=list(f['candidate'].shape),source_windows=int((f['source_doc_index']>=0).sum()))
        else:out=self.model(**f,freeze=freeze)
        if self.mode=='frozen_old_read':
            for di,doc in enumerate(docs):
                if doc['doc_id'] in self.old_reads: continue
                self.old_reads[doc['doc_id']]=[dict(tokens=f['tokens'][si].clone(),mask=f['token_mask'][si].clone(),link=f['links'][si].clone(),score=out['source'][si].clone(),value=out['source_values'][si].clone(),special=out['source_special'][si].clone(),ids={v:[c['candidate_id'] for c in domain[v]] for v in vars}) for si in (f['source_doc_index']==di).nonzero().flatten().tolist()]
        beams=decode(out['unary'],out['pair'],f['valid'],beam=self.config['rebind']['joint_beam'],constraint=lambda a:constraint_penalty(graph,domain,a))
        assignments=[{v:domain[v][a[i]]['surface'] for i,v in enumerate(vars)} for _,a in beams]
        candidate_assignments=[{v:domain[v][a[i]]['candidate_id'] for i,v in enumerate(vars)} for _,a in beams]
        query_beams=sorted(beams,key=lambda item:-(item[0]+.2*sum(float(out['frontier'][i,ci]) for i,ci in enumerate(item[1]))))
        query_assignments=[{v:domain[v][a[i]]['surface'] for i,v in enumerate(vars)} for _,a in query_beams]
        current={v:domain[v][beams[0][1][i]]['candidate_id'] for i,v in enumerate(vars)} if beams else {}
        revisions=[dict(variable=v,before=self.previous[v],after=c,verified=False) for v,c in current.items() if v in self.previous and self.previous[v]!=c]
        self.previous=current
        source_indices=f.get('source_doc_index',torch.arange(len(docs),device=self.device))
        source_scores={docs[int(source_indices[s])]['doc_id']+f':window{s}':[out['source'][s,i,:len(domain[v])].cpu().tolist() for i,v in enumerate(vars)] for s in range(len(out['source'])) if int(source_indices[s])>=0}
        result=dict(assignments=assignments,candidate_assignments=candidate_assignments,query_assignments=query_assignments,source_scores=source_scores,revisions=revisions,observed_slots=[],answer_ready=False,status='inferred_or_unresolved',candidate_ids=current)
        if activity is not None:
            no_pair=decode(out['unary'],torch.zeros_like(out['pair']),f['valid'],beam=self.config['rebind']['joint_beam'],constraint=lambda a:constraint_penalty(graph,domain,a))
            activity['pair_potential_changes_top_decode']=bool(beams and no_pair and beams[0][1]!=no_pair[0][1]);result['operator_activity']=activity
        return result

def prepare_natural(config,limit=None):
    """Offline train-only aligner; model forward sees only each displayed prefix."""
    from .data import read_rows,contexts,document_id
    from .schema import InferenceExample,QuestionGraph
    from .proposal import Generator,propose
    from .retrieval import E5
    from .audit import append
    import collections,re,time
    root=pathlib.Path(config['paths']['workdir']); data=pathlib.Path(config['paths']['data_root']); directory=data/'natural_transitions'; directory.mkdir(exist_ok=True)
    generator=Generator(config); encoder=E5(config['models']['retriever_path'],device=resolve_device(config,'retriever')); builder=FeatureBuilder(encoder,config); audit=collections.Counter(); records=[]
    def terms(s): return {w[:5] for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3}
    for split in ['train','dev']:
        rows=list(read_rows(data/'private/2wiki'/f'{split}.jsonl'))
        if limit: rows=rows[:limit]
        def process_row(item):
            ri,record=item
            example=InferenceExample(record['qid'],record['raw_record']['question']); graph=None; registry=None
            docs=[dict(doc_id=document_id(title,text),parent_doc_id=document_id(title,text),title=title,text=text,offsets=[0,len(text)],sentence_offsets=pos) for title,text,pos in contexts(record['raw_record'],'2wiki')]
            docs.sort(key=lambda d:d['doc_id']); previous=None
            generator.context=dict(qid=example.qid,split=split,protocol='training_controlled_prefix',source_hash=digest(root/'manifests/sources.lock.json'))
            for count in sorted(set([min(3,len(docs)),min(6,len(docs)),len(docs)])):
                try:
                    graph,registry,proposal=propose(generator,example,docs[:count],graph,registry,count)
                    visible=proposal['visible']; f=builder.build(example.question,graph,registry,visible,proposal,dict(qid=example.qid,split=split)); domain=registry.snapshot(); vars=list(domain); n,k=f['valid'].shape
                    allowed=torch.zeros_like(f['valid']); mask=torch.zeros(n,dtype=torch.bool); pairallowed=torch.zeros(n,n,k,k,dtype=torch.bool); pairmask=torch.zeros(n,n,dtype=torch.bool)
                    support_docs=[d for d in visible if d['parent_doc_id'] in record['supporting_doc_ids']]
                    text='\n'.join(d['title']+'\n'+d['text'] for d in support_docs)
                    label_provenance=[]
                    for i,var in enumerate(graph.variables):
                        matching=[]
                        if var.get('is_answer'):
                            matching=[a for a in record['answers'] if a in text and a not in ['yes','no']]
                        else:
                            for subject,predicate,obj in record['evidences']:
                                if subject in text and obj in text and terms(predicate)&terms(var['description']): matching.append(obj)
                        for a,candidate in enumerate(domain[vars[i]]):
                            if candidate['surface'] in matching:
                                allowed[i,a]=True
                                for doc in support_docs:
                                    position=doc['text'].find(candidate['surface'])
                                    if position>=0: label_provenance.append(dict(variable=vars[i],candidate_id=candidate['candidate_id'],doc_id=doc['doc_id'],char_start=doc['offsets'][0]+position,char_end=doc['offsets'][0]+position+len(candidate['surface']),quote=candidate['surface'],verifier='literal_span_plus_weak_role_alignment',review_status='rule_aligned_unreviewed'))
                        mask[i]=allowed[i].any(); audit['valid_unary_labels']+=int(mask[i])
                    # Reliable pair supervision only for a directed proposed relation aligned to a visible official triple.
                    for slot in graph.slots:
                        arguments=slot['ordered_arguments']
                        if len(arguments)!=2 or any(v not in vars for v in arguments): continue
                        i,j=[vars.index(v) for v in arguments]
                        if i==j: continue
                        for subject,predicate,obj in record['evidences']:
                            if subject not in text or obj not in text or not terms(predicate)&terms(slot['relation_text']): continue
                            for a,ca in enumerate(domain[vars[i]]):
                                for b,cb in enumerate(domain[vars[j]]):
                                    if ca['surface']==subject and cb['surface']==obj:
                                        pairallowed[i,j,a,b]=True; pairmask[i,j]=True
                    audit['valid_pair_labels']+=int(pairmask.sum()); audit['prefixes']+=1
                    payload=dict(label_provenance=label_provenance,features=f,labels=dict(unary=dict(allowed=allowed,valid=f['valid'],mask=mask),frontier=dict(allowed=allowed,valid=f['valid'],mask=mask),pair=dict(allowed=pairallowed,valid=f['valid'][:,None,:,None]&f['valid'][None,:,None,:],mask=pairmask)),qid=example.qid,group_id=record['grouping_metadata']['group_id'],split=split,change_type='support_addition_or_unresolved',source='official_training_labels_aligned_to_current_raw_prefix',review_status='rule_aligned_unreviewed',confidence='weak',model_inputs=dict(question=example.question,graph=graph.__dict__,candidates=domain,documents=visible),previous_prefix=previous)
                    ident=digest([example.qid,count,config,domain,visible]); file=directory/(ident+'.pt'); torch.save(payload,file); previous=str(file)
                    records.append(dict(path=str(file),qid=example.qid,split=split,valid_unary=int(mask.sum()),valid_pair=int(pairmask.sum())))
                except (ValueError,KeyError,TypeError) as e:
                    audit['proposal_failures']+=1; append(root/'reports/failure_cases.jsonl',dict(stage='natural_transition',qid=example.qid,prefix_count=count,error=str(e)))
            if ri%8==0: print('natural transitions',split,ri,len(rows),dict(audit),flush=True)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("REBIND_WORKERS","8"))) as pool: list(pool.map(process_row,enumerate(rows)))
    audit.update(valid_unary_labels=sum(r['valid_unary'] for r in records),valid_pair_labels=sum(r['valid_pair'] for r in records),prefixes=len(records))
    write(directory/'manifest.json',dict(config_hash=digest(config),records=records,counts=dict(audit),revision_labels=0,source_scope_labels=0,limitations=['Surface/relation alignment is partial supervision, not human validated role/scope annotation.','No future evidence or gold graph is passed to the model.']))
    return dict(audit)

def train_natural(config,seeds=None):
    import random,time,csv
    from .train import transition_loss
    from .audit import append,Blocked
    root=pathlib.Path(config['paths']['workdir']); data=pathlib.Path(config['paths']['data_root']); manifestfile=data/'natural_transitions/manifest.json'
    if not manifestfile.exists(): raise Blocked('Natural transition manifest missing')
    manifest=json.loads(manifestfile.read_text()); valid=[r for r in manifest['records'] if r['valid_unary']+r['valid_pair']>0]; rows={s:[r for r in valid if r['split']==s] for s in ['train','dev']}
    if not rows['train'] or not rows['dev']: raise Blocked('No informative train/dev natural supervision')
    loaded={row['path']:torch.load(row['path'],weights_only=True,mmap=True) for row in valid}
    device=resolve_device(config,'train'); results=[]
    for method in ['bp_rebind','rebind','independent_binding','no_revision_loss']:
        for seed in seeds or config['train']['seeds']:
            rows['train']=sorted(rows['train'],key=lambda r:(r['qid'],r['path']))
            torch.manual_seed(seed); model=ReBindModule(input_dim=next(iter(loaded.values()))['features']['candidate'].shape[-1],d=config['rebind']['hidden_dim'],layers=config['rebind']['layers'],mode='rebind' if method=='no_revision_loss' else method).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=config['train']['learning_rate'],weight_decay=config['train']['weight_decay']); best=float('inf'); directory=data/'checkpoints'/method/str(seed); directory.mkdir(parents=True,exist_ok=True)
            for epoch in range(config['train']['epochs_initial']):
                rng=random.Random(seed+epoch); rng.shuffle(rows['train']); stats={}
                for split in ['train','dev']:
                    model.train(split=='train'); losses=[]; count=0
                    opt.zero_grad(set_to_none=True)
                    accumulation=config['train']['gradient_accumulation']
                    for ri,row in enumerate(rows[split]):
                        ex=loaded[row['path']]; f={k:v.to(device) for k,v in ex['features'].items()}; labels={field:{k:v.to(device) for k,v in item.items()} for field,item in ex['labels'].items()}
                        with torch.set_grad_enabled(split=='train'):
                            output=model(**f); loss=.25*transition_loss(output,output,{'after':labels},revision_weight=0 if method=='no_revision_loss' else .5)
                            if not torch.isfinite(loss): raise ValueError('Nonfinite natural training loss')
                            if split=='train':
                                (loss/min(accumulation,len(rows[split])-(ri//accumulation)*accumulation)).backward()
                                if (ri+1)%accumulation==0 or ri+1==len(rows[split]):
                                    torch.nn.utils.clip_grad_norm_(model.parameters(),config['train']['gradient_clip']); opt.step(); opt.zero_grad(set_to_none=True)
                        losses.append(float(loss.detach())); count+=1
                    stats[split]=sum(losses)/len(losses)
                ckpt=dict(model_code_hash=digest({name:digest(root/'src/rebind_mvp'/name) for name in ['source_reader.py','pair_update.py','bp.py','train.py']}),model=model.state_dict(),optimizer=opt.state_dict(),method=method,seed=seed,epoch=epoch,config_hash=digest(config),data_hash=digest(manifestfile),scope='natural_only_partial_binding_supervision')
                torch.save(ckpt,directory/'last.pt')
                if stats['dev']<best: best=stats['dev']; torch.save(ckpt,directory/'best.pt')
                row=dict(data_hash=digest(manifestfile),base_train_questions=len({r['qid'] for r in rows['train']}),method=method,seed=seed,epoch=epoch,train_loss=stats['train'],dev_loss=stats['dev'],train_prefixes=len(rows['train']),dev_prefixes=len(rows['dev']));append(root/'reports/natural_learning_curves.jsonl',row);print(row,flush=True)
            results.append(dict(method=method,seed=seed,best_dev_loss=best,checkpoint=str(directory/'best.pt')))
    summary=dict(results=results,data_hash=digest(manifestfile),config_hash=digest(config),revision_labels=manifest['revision_labels'],source_scope_labels=manifest['source_scope_labels'],scope='natural-only partial binding supervision',limitations=manifest['limitations'])
    for seed in sorted({r['seed'] for r in results}): write(root/'reports'/f'natural_training_seed{seed}.json',dict(summary,results=[r for r in results if r['seed']==seed]))
    if seeds is None: write(root/'reports/natural_training.json',summary)
    return summary
