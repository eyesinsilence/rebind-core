from .runtime import resolve_device,encoder_dimension,peak_memory
"""Query replay/proposal isolation and development-only frontier interventions."""
import argparse, collections, copy, csv, inspect, json, os, pathlib, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .audit import digest, write, append
from .schema import InferenceExample
from .evaluate import run_loop
from .mquake import EditRetriever, official_function


def policy(kind, generator, common):
    def apply(ex, graph, state, queries, raw, round_index, slot_id, native_query):
        audit = dict(kind=kind, native_slot=slot_id, native_query=native_query)
        try:
            if kind == 'hybrid':
                prompt = 'Read the question and raw updates. Propose one next retrieval query. The supplied inferred bindings are uncertain, may be wrong, and may be revised; never treat them as source evidence. Return JSON {"next_query":"specific next search"}. Do not answer the question.\nQuestion: '+ex.question+'\nPrevious queries: '+json.dumps(queries)+'\nUncertain bindings: '+json.dumps(state.get('assignments',[])[:2])+'\nDocuments:\n'+raw
                reply = generator.json(prompt, max_tokens=256)
                query = reply.get('next_query')
            else:
                if kind == 'common_slot':
                    if round_index+1 >= len(common['trace']):
                        return slot_id, native_query, dict(audit, fallback='COMMON schedule exhausted; native policy')
                    slots = {s['slot_id']: s for s in graph.slots}
                    prompt = 'Map this recorded reference search to ONE relation in the supplied predicted question graph. Return JSON {"slot_id":"ID or NONE"}. Match the relation being queried, not the entity names. Do not propose arguments or a new relation.\nRelations: '+json.dumps([dict(slot_id=s['slot_id'],relation=s['relation_text']) for s in graph.slots])+'\nReference search: '+common['trace'][round_index+1]['query']
                    reply = generator.json(prompt,max_tokens=64,schema={'type':'object','properties':{'slot_id':{'type':'string','enum':list(slots)+['NONE']}},'required':['slot_id']})
                    slot_id = reply.get('slot_id')
                    audit.update(reference_query=common['trace'][round_index+1]['query'], mapped_slot=slot_id)
                    if slot_id not in slots:
                        return audit['native_slot'], native_query, dict(audit, fallback='unmapped reference relation; native policy')
                    slot = slots[slot_id]
                    query = None
                    for assignment in state.get('query_assignments',state.get('assignments',[])):
                        args = [assignment.get(v,'UNKNOWN') for v in slot['ordered_arguments']]
                        candidate = slot['relation_text']+' '+' '.join(a for a in args if a!='UNKNOWN')
                        if any(a!='UNKNOWN' for a in args) and candidate not in queries:
                            query=candidate; audit['arguments']=args; break
                else:
                    slot = next((s for s in graph.slots if s['slot_id']==slot_id),None)
                    if slot is None or native_query is None:
                        return slot_id,native_query,dict(audit,fallback='no native target')
                    args = next([a.get(v,'UNKNOWN') for v in slot['ordered_arguments']] for a in state.get('query_assignments',state.get('assignments',[])) if slot['relation_text']+' '+' '.join(a.get(v,'UNKNOWN') for v in slot['ordered_arguments'] if a.get(v,'UNKNOWN')!='UNKNOWN') == native_query)
                    target=dict(relation=slot['relation_text'],ordered_arguments=args)
                    prompt='Turn the supplied relation and ordered arguments into one short natural-language search query. UNKNOWN is an unbound endpoint, not a literal name. Preserve EVERY known argument exactly, preserve relation and direction, and add no new entities, dates, or facts. Do not solve the relation. Return JSON {"next_query":"query"}.\nTarget: '+json.dumps(target)
                    reply=generator.json(prompt,max_tokens=128)
                    query=reply.get('next_query');audit['target']=target
                    if not isinstance(query,str) or any(a!='UNKNOWN' and a.casefold() not in query.casefold() for a in args):
                        raise ValueError('verbalizer dropped a bound argument')
            if not isinstance(query,str) or not query.strip() or query.strip() in queries:
                raise ValueError('empty or repeated policy query')
            audit['emitted_query']=query.strip()
            return slot_id,query.strip(),audit
        except (ValueError,KeyError,TypeError,StopIteration) as error:
            return audit['native_slot'],native_query,dict(audit,fallback=repr(error))
    return apply


def run(c, scope):
    import torch
    from .proposal import Generator
    from .retrieval import E5
    from .transitions import NeuralState
    root=pathlib.Path(c['paths']['workdir']); prior=pathlib.Path(c['frontier_diagnostics']['prior']); parent=pathlib.Path(c['diagnostics']['parent'])
    folder=root/'runs'/('frontier_'+scope);folder.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'data/adapt/split.json').read_text())
    ids=set(split['split']['test'] if scope=='test' else split['split']['dev'][:4 if scope=='smoke' else 32])
    rows=[r for r in json.loads((root/'data/public/T.json').read_text()) if r['case_id'] in ids and (scope!='smoke' or r['variant']==0)]
    selections=json.loads((parent/'manifests/adapt_qa_lock.json').read_text())['identity']['all_trained_selections']
    arms=['open_native','open_matched_rounds','query_native','query_legacy_proposals'] if scope!='dev' else ['open_native']
    if scope!='test':arms+=['common_slot','verbalize','no_frontier_score','hybrid']
    seeds=[17,29,43] if scope=='test' else [17]
    references={}
    if scope in ['test','smoke']:
        for row in rows:
            commonpath=(parent/'runs/adapt_qa'/(row['qid']+'_ircot_common.json')) if scope=='test' else prior/'runs/diagnostics_smoke'/(row['qid']+'_open_common_reference.json')
            references[str(commonpath)]=digest(commonpath)
            cachepath=prior/('runs/diagnostics_'+('test' if scope=='test' else 'smoke'))/(row['qid']+'_proposals.pt')
            references[str(cachepath)]=digest(cachepath)
    identity=dict(config=digest(c),scope=scope,rows=digest(rows),sources={str(p.relative_to(root)):digest(p) for p in (root/'src').rglob('*.py') if p.name!='frontier_diagnostics.py'},runner=digest(inspect.getsource(run)),policy=digest(inspect.getsource(policy)),references=references,checkpoints={str(selections[f'rebind_s{s}']['path']):digest(pathlib.Path(selections[f'rebind_s{s}']['path'])) for s in seeds},data={str(root/'data'/p):digest(root/'data'/p) for p in ['public/T.json','adapt/split.json','memory/T.json','memory/T_e5.npy']},arms=arms,seeds=seeds)
    h=digest(identity);lock=root/'manifests'/('frontier_'+scope+'_lock.json')
    if lock.exists():assert json.loads(lock.read_text())['identity']==identity,'Identity changed; do not reuse predictions'
    else:write(lock,dict(identity=identity,hash=h,time=time.time()))
    generator=Generator(c);encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever'));docs=json.loads((root/'data/memory/T.json').read_text());vectors=np.load(root/'data/memory/T_e5.npy')
    def task(row):
        ex=InferenceExample(row['qid'],row['question']);outputs=[]
        generator.context=dict(qid=ex.qid,split='frontier_'+scope,source_hash=identity['data'][str(root/'data/memory/T.json')])
        commonfile=folder/(ex.qid+'_common_reference.json')
        if commonfile.exists():common=json.loads(commonfile.read_text());assert common['identity_hash']==h
        elif scope in ['test','smoke']:
            p=next(pathlib.Path(p) for p in references if p.endswith(ex.qid+('_ircot_common.json' if scope=='test' else '_open_common_reference.json')))
            common=json.loads(p.read_text());common.update(method='common_reference',identity_hash=h,imported=True,parent_artifact=str(p),parent_artifact_hash=references[str(p)]);write(commonfile,common)
        else:
            common=run_loop(ex,EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k']),generator,c,'ircot_common')
            common.update(case_id=row['case_id'],variant=row['variant'],method='common_reference',identity_hash=h,imported=False);write(commonfile,common)
        outputs.append(common)
        for arm in arms:
            for seed in seeds:
                name=f'{arm}_s{seed}';path=folder/(ex.qid+'_'+name+'.json')
                if path.exists():r=json.loads(path.read_text());assert r['identity_hash']==h;outputs.append(r);continue
                start=time.time();first=len(generator.calls);retriever=EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k']);checkpoint=selections[f'rebind_s{seed}'];neural=NeuralState(encoder,c,pathlib.Path(checkpoint['path']),'rebind',split='frontier_'+scope)
                kwargs={};cache=None
                if arm.startswith('query_'):kwargs['query_schedule']=[t['query'] for t in common['trace']]
                if arm=='open_matched_rounds':kwargs['force_rounds']=len(common['trace'])
                if arm=='query_legacy_proposals':
                    p=prior/('runs/diagnostics_'+('test' if scope=='test' else 'smoke'))/(ex.qid+'_proposals.pt');cache=copy.deepcopy(torch.load(p,weights_only=False,map_location='cpu'));kwargs['proposal_cache']=cache
                if arm in ['common_slot','verbalize','hybrid']:kwargs['query_policy']=policy(arm,generator,common)
                if arm=='no_frontier_score':kwargs['use_query_head']=False
                try:r=run_loop(ex,retriever,generator,c,'rebind',neural,**kwargs)
                except Exception as error:r=dict(answer='',trace=[],fallback=1,eval_status='runtime_failure',error=repr(error),traceback=traceback.format_exc())
                calls=generator.calls[first:];r.update(qid=ex.qid,case_id=row['case_id'],variant=row['variant'],arm=arm,method=name,seed=seed,identity_hash=h,checkpoint_hash=checkpoint['sha256'],imported=False,seconds=time.time()-start,actual_retriever_calls=retriever.calls,input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls),actual_requests=sum(not x['cache_hit'] for x in calls),max_prompt_tokens=max([x['input_tokens'] for x in calls] or [0]))
                if arm.startswith('query_'):
                    strip=lambda ts:[dict(query=t['query'],docs=[{k:v for k,v in d.items() if k!='score'} for d in t['retrieved_documents']],visible=t['visible_spans']) for t in ts]
                    r['replay_audit']=dict(query_match=[t['query'] for t in r['trace']]==[t['query'] for t in common['trace']],raw_match=strip(r['trace'])==strip(common['trace']),actual_search_calls=len(retriever.calls))
                write(path,r);outputs.append(r);print(scope,ex.qid,name,r['eval_status'],round(r['seconds'],2),flush=True);del neural
        return outputs
    with ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS','48'))) as pool:results=[r for batch in pool.map(task,rows) for r in batch]
    status=dict(scope=scope,records=len(results),expected=len(rows)*(1+len(arms)*len(seeds)),computed=sum(not r['imported'] for r in results),imported=sum(r['imported'] for r in results),failures=sum(r['eval_status']!='ok' for r in results),hash=h)
    write(folder/'summary.json',status);return status


def report(c,scope):
    root=pathlib.Path(c['paths']['workdir']);folder=root/'runs'/('frontier_'+scope);lock=json.loads((root/'manifests'/('frontier_'+scope+'_lock.json')).read_text());identity=lock['identity'];records=[json.loads(p.read_text()) for p in folder.glob('T_*.json')]
    gold=json.loads((root/'data/private/T.json').read_text());check=official_function(root,'check_answer');groups={r['case_id']:r['group'] for r in json.loads((root/'data/adapt/samples.json').read_text())}
    rows=collections.defaultdict(list);scores={};table=[];qgroups={};failures=[]
    for r in records:
        assert r['identity_hash']==lock['hash'];r['correct']=bool(check(True,gold[str(r['case_id'])],r['answer']));rows[r['method']].append(r);qgroups[r['qid']]=groups[r['case_id']]
        if r['eval_status']!='ok' or r.get('fallback') or any(t.get('query_policy',{}).get('fallback') for t in r['trace']):failures.append({k:r.get(k) for k in ['qid','method','eval_status','fallback','error']}|{'policy_fallbacks':[t['query_policy'] for t in r['trace'] if t.get('query_policy',{}).get('fallback')]})
    variants=1 if scope=='smoke' else 3
    for method,rr in sorted(rows.items()):
        cases=collections.defaultdict(list)
        for r in rr:cases[r['case_id']].append(r['correct'])
        scores[method]=dict(question={r['qid']:float(r['correct']) for r in rr},case={q:float(any(rs)) for q,rs in cases.items() if len(rs)==variants})
        table.append(dict(method=method,questions=len(rr),complete_cases=len(scores[method]['case']),case_accuracy=float(np.mean(list(scores[method]['case'].values()))) if scores[method]['case'] else None,question_accuracy=float(np.mean([r['correct'] for r in rr])),failures=sum(r['eval_status']!='ok' for r in rr),fallback_questions=sum(bool(r.get('fallback')) for r in rr),policy_fallback_questions=sum(any(t.get('query_policy',{}).get('fallback') for t in r['trace']) for r in rr),mean_retrievals=float(np.mean([len(r['trace']) for r in rr])),input_tokens=sum(r.get('input_tokens',r.get('llm_input_tokens',0)) for r in rr if not r['imported']),max_prompt_tokens=max(r.get('max_prompt_tokens',0) for r in rr)))
    for arm in identity['arms']:
        names=[f'{arm}_s{s}' for s in identity['seeds']]
        if all(n in scores for n in names):scores[arm]={metric:{q:float(np.mean([scores[n][metric][q] for n in names])) for q in set.intersection(*(set(scores[n][metric]) for n in names))} for metric in ['case','question']}
    comparisons=[('query_native','open_native'),('query_native','open_matched_rounds'),('open_matched_rounds','open_native'),('query_legacy_proposals','query_native'),('query_native','common_reference')]+[(arm,'open_native') for arm in ['common_slot','verbalize','no_frontier_score','hybrid']]
    pairs=[]
    for treatment,control in comparisons:
        if treatment not in scores or control not in scores:continue
        for metric in ['case','question']:
            aa=scores[treatment][metric];bb=scores[control][metric];gg=collections.defaultdict(list)
            for q in sorted(aa.keys()&bb.keys()):gg[groups[q] if metric=='case' else qgroups[q]].append(aa[q]-bb[q])
            if not gg:continue
            sizes=np.array([len(v) for v in gg.values()]);sums=np.array([sum(v) for v in gg.values()]);draw=np.random.default_rng(612).integers(0,len(gg),(2000,len(gg)));boot=sums[draw].sum(1)/sizes[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);pairs.append(dict(treatment=treatment,control=control,metric=metric,n=int(sizes.sum()),groups=len(gg),difference=float(sums.sum()/sizes.sum()),ci_low=float(lo),ci_high=float(hi)))
    prefix=root/'reports'/('frontier_'+scope)
    for name,rs in [('results',table),('paired',pairs)]:
        with pathlib.Path(str(prefix)+'_'+name+'.csv').open('w') as f:
            if rs:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    write(str(prefix)+'_failures.json',failures)
    with pathlib.Path(str(prefix)+'_predictions.jsonl').open('w') as f:
        for r in records:f.write(json.dumps({k:r.get(k) for k in ['qid','case_id','variant','method','seed','answer','correct','eval_status','fallback','identity_hash','imported','replay_audit']})+'\n')
    unchanged=digest(c)==identity['config'] and digest(inspect.getsource(run))==identity['runner'] and digest(inspect.getsource(policy))==identity['policy']
    unchanged=unchanged and all(digest(root/p)==sha for p,sha in identity['sources'].items()) and all(digest(pathlib.Path(p))==sha for collection in ['references','checkpoints','data'] for p,sha in identity[collection].items())
    split=json.loads((root/'data/adapt/split.json').read_text());expected_questions=(len(split['split']['test'])*3 if scope=='test' else 4 if scope=='smoke' else 32*3);expected=expected_questions*(1+len(identity['arms'])*len(identity['seeds']))
    replay=[r for r in records if r.get('arm','').startswith('query_')];allowed={r['qid']:set(r['allowed_doc_ids']) for r in json.loads((root/'data/public/T.json').read_text())}
    bad_docs=[(r['qid'],r['method']) for r in records if any(set(t.get('retrieved_ids',[]))-allowed[r['qid']] for t in r['trace'])]
    status=dict(status='complete' if len(records)==expected and unchanged else 'in_progress',records=len(records),expected=expected,inference_unchanged=unchanged,hash=lock['hash'],replay_questions=len(replay),replay_query_mismatches=sum(not r.get('replay_audit',{}).get('query_match') for r in replay),replay_raw_mismatches=sum(not r.get('replay_audit',{}).get('raw_match') for r in replay),illegal_documents=bad_docs,failures=sum(r['eval_status']!='ok' for r in records))
    write(str(prefix)+'_completion.json',status)
    lines=['# Frontier diagnostics: '+scope,'',json.dumps(status),'','Exploratory. Query replay performs real retrieval; native proposals remain independent of the legacy shared proposal cache. Matched-rounds control isolates schedule length. COMMON relation mapping is an approximate diagnostic, not gold slots. Frontier repairs run on development cases only; do not tune them on the test replay results.','', '| Method | Questions | Case accuracy | Question accuracy | Final failures |','|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['method']} | {r['questions']} | {r['case_accuracy']} | {r['question_accuracy']:.4f} | {r['failures']} |")
    (root/'reports'/('FRONTIER_'+scope.upper()+'.md')).write_text('\n'.join(lines)+'\n')
    return status


if __name__=='__main__':
    import yaml,sys
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run','report']);parser.add_argument('--scope',choices=['smoke','dev','test'],required=True);args=parser.parse_args();c=yaml.safe_load(pathlib.Path('configs/frontier.yaml').read_text());start=time.time();status='ok'
    try:print(json.dumps(run(c,args.scope) if args.phase=='run' else report(c,args.scope)),flush=True)
    except BaseException:status='failed';raise
    finally:append(pathlib.Path(c['paths']['workdir'])/'runs/frontier/commands.jsonl',dict(argv=sys.argv,start=start,seconds=time.time()-start,status=status,code_hash=digest(pathlib.Path(__file__))))
