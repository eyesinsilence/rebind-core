from .runtime import resolve_device,encoder_dimension,peak_memory
import collections,csv,importlib.util,json,pathlib,sys,time,os
import numpy as np
import torch
from .audit import digest,write,append,execute,Blocked
from .data import read_rows
from .schema import InferenceExample

def run(name,c,args):
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); reports=root/'reports'
    if name=='audit-sources':
        sources=json.loads((root/'manifests/sources.lock.json').read_text()); lines=['# Related work audit','Official repositories cloned and commits locked. Narrow architecture novelty remains unverified; external reproduction and strong comparisons are required.','']
        for source in sources: lines.append(f"- [{source['name']}]({source['url']}): `{source.get('commit','unavailable')}`; clone exit {source['exit_code']}.")
        lines+=['','Mechanisms to control: IRCoT interleaves reasoning/retrieval; TaSR-RAG already uses typed triples and binding tables; ReAgent uses reversible multi-agent reasoning; EoG revisits beliefs using graph propagation; PyRAG uses executable variables and feedback retrieval; joint entity disambiguation and DyGIE already perform joint graph inference. These high-level overlaps do not establish identity or novelty of candidate-level incremental source reinterpretation.','', 'Paper pages checked:']
        for url in ['2603.09341','2503.06951','2601.17915','2605.12975','1704.04920','1904.03296']: lines.append('- https://arxiv.org/abs/'+url)
        if not (reports/'related_work_audit.md').exists(): (reports/'related_work_audit.md').write_text('\n'.join(lines)+'\n')
        result={}
        from .adapters.reagent import check
        try: result['reagent']=check(root)
        except Blocked as e: result['reagent']=dict(status='blocked',reason=str(e))
        # Only import official PyRAG components; generated code smoke uses isolated executor.
        code="import sys; sys.path.insert(0,'upstream/PyRAG'); from pyrag.runner import RAGProgramRunner; print('official controller import OK')"
        result['pyrag_import']=execute([sys.executable,'-c',code],root,'pyrag_official_import',timeout=30)
        from .adapters.pyrag import SandboxedExecutor
        result['pyrag_sandbox']=SandboxedExecutor().execute("final_answer=answer('question',retrieve('query'))",lambda q:['raw'],lambda q,d:'sandbox-ok',[])
        write(reports/'external_audit.json',result); return result
    if name=='audit-data':
        manifest=json.loads((root/'manifests/splits.json').read_text()); audit=json.loads((reports/'data_audit.json').read_text())
        for ds in c['retrieval']['datasets']:
            sets={}
            for split,record in manifest['datasets'][ds].items():
                p=data/'public'/ds/f'{split}.jsonl'; assert digest(p)==record['public_sha256']
                rows=list(read_rows(p)); assert all(set(row)=={'qid','question','visible_documents'} for row in rows); assert all(not r['visible_documents'] for r in rows)
                sets[split]={r['qid'] for r in rows}
            assert not sets['train']&sets['dev'] and not sets['train']&sets['eval'] and not sets['dev']&sets['eval']
            audit[ds]['public_label_allowlist_verified']=True
        write(reports/'data_audit.json',audit); return audit
    if name=='audit-retrieval':
        from .retrieval import Retriever,E5
        encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever')); result={}
        for ds in getattr(args,'datasets',None) or c['retrieval']['datasets']:
            r=Retriever(c,ds,encoder); rows=list(read_rows(data/'public'/ds/'dev.jsonl'))[:5]; checks=[]
            for row in rows:
                docs=r.search(row['question']); v=encoder.encode([row['question']],'query')[0]
                score=np.sum(r.vectors*v[None,:],axis=1); order=np.argsort(-score,kind='stable')[:r.top_k]
                checks.append(dict(qid=row['qid'],same_ids=[r.docs[i]['doc_id'] for i in order]==[d['doc_id'] for d in docs],max_score_error=max(abs(float(score[i])-d['score']) for i,d in zip(order,docs))))
            assert all(x['same_ids'] for x in checks)
            norm=np.linalg.norm(r.vectors[:4096],axis=1); assert np.max(abs(norm-1))<1e-5
            result[ds]=dict(checks=checks,unit_norm_max_error=float(np.max(abs(norm-1))),index_manifest=json.loads((data/'indexes'/ds/'manifest.json').read_text()))
        write(reports/'retrieval_audit.json',result); return result
    if name in ['diagnose','smoke','evaluate']:
        return run_questions(name,c,args)
    if name=='prepare-transitions':
        if not (root/'manifests/diagnose.json').exists(): raise Blocked('Natural diagnosis must run before natural training transitions; independent micro fixtures are already separate.')
        from .transitions import prepare_natural
        return prepare_natural(c,limit=getattr(args,'limit',None) or c.get('execution',{}).get('transition_pilot_questions'))
    if name=='train':
        from .transitions import train_natural
        return train_natural(c,seeds=getattr(args,'seeds',None))
    if name=='intervene':
        from .interventions import run_interventions
        return run_interventions(c)
    if name=='report':
        return report(c)
    raise ValueError(name)

def official_scores(root,ds,prediction,record):
    if ds=='2wiki':
        path=root/'upstream/2wikimultihop/2wikimultihop_evaluate_v1.1.py'
        spec=importlib.util.spec_from_file_location('wiki_eval',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        em=max(float(module.exact_match_score(prediction,a)) for a in record['answers']); f1=max(float(module.f1_score(prediction,a)[0]) for a in record['answers'])
    else:
        sys.path.insert(0,str(root/'upstream/musique'))
        from metrics.answer import AnswerMetric
        m=AnswerMetric(); m(prediction,record['answers']); em,f1=m.get_metric()
    return dict(em=em,f1=f1)

def run_questions(stage,c,args):
    from .proposal import Generator
    from .retrieval import E5,Retriever
    from .evaluate import run_loop
    from .transitions import NeuralState
    from .adapters.flashrag_ircot import native_run
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); split='eval' if stage=='evaluate' else 'dev'; protocol=getattr(args,'protocol','open')
    import inspect
    inference_files=[p for p in (root/'src/rebind_mvp').rglob('*.py') if p.name not in ['stages.py','cli.py','data.py','audit.py','train.py','diagnostics.py','interventions.py']]
    pipeline_hash=digest(dict(files={str(p.relative_to(root/'src')):digest(p) for p in inference_files},runner=inspect.getsource(run_questions)))
    phase_dir=root/'runs'/('natural_'+stage+'_'+protocol); phase_dir.mkdir(exist_ok=True); generator=Generator(c); encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever'))
    limit=getattr(args,'limit',None) or (c.get('execution',{}).get('diagnose_questions',200) if stage=='diagnose' else c.get('execution',{}).get('smoke_questions',16))
    if stage=='evaluate': limit=None
    methods=['ircot_native','ircot_common','json_rebind'] if stage=='diagnose' else c['methods']['internal']
    if stage=='smoke':
        from .source_reader import ReBindModule
        for mode in ['bp_rebind','rebind']:
            initial=data/'smoke_initialization'/mode/'initial.pt'; initial.parent.mkdir(parents=True,exist_ok=True)
            if not initial.exists():
                torch.manual_seed(17); model=ReBindModule(input_dim=encoder_dimension(encoder),d=c['rebind']['hidden_dim'],layers=c['rebind']['layers'],mode=mode); torch.save(dict(model=model.state_dict(),scope='untrained_smoke'),initial)
    if protocol=='fixed': methods=['direct_reader','json_rebind','bp_rebind','rebind']
    if getattr(args,'methods',None): methods=args.methods
    allowed={'ircot_native','ircot_common','direct_reader','json_rebind','bp_rebind','rebind','independent_binding','no_revision_loss','frozen_old_read','frozen_query_replay','pyrag'}
    if set(methods)-allowed: raise Blocked('Unsupported or blocked method: '+','.join(sorted(set(methods)-allowed))+'; see reports/external_audit.json for ReAgent')
    if stage=='evaluate':
        lockfile=root/'manifests/benchmark_lock.json'; locked=dict(config_hash=digest(c),pipeline_hash=pipeline_hash,split_hash=digest(root/'manifests/splits.json'),checkpoint_hashes={str(p.relative_to(data)):digest(p) for p in sorted((data/'checkpoints').glob('*/*/best.pt'))})
        if lockfile.exists():
            previous=json.loads(lockfile.read_text())
            if any(previous[k]!=v for k,v in locked.items()): raise Blocked('Sealed evaluation configuration/checkpoint changed; preserve old protocol and explicitly audit the change.')
        else:
            summaries=[json.loads(p.read_text()) for p in (root/'reports').glob('natural_training_seed*.json')]
            selected={(r['method'],r['seed']) for summary in summaries if summary.get('data_hash')==digest(data/'natural_transitions/manifest.json') and summary.get('config_hash')==digest(c) for r in summary['results']}
            expected={(m,seed) for m in ['bp_rebind','rebind','independent_binding','no_revision_loss'] for seed in c['train']['seeds']}
            if selected!=expected: raise Blocked('All full-data train/dev checkpoint selections must precede sealed evaluation')
            write(lockfile,dict(locked,locked_at=time.time(),selection='official-dev eval IDs were fixed before outputs; checkpoints selected on training-side dev'))
    results=[]; blocked=[]
    for ds in getattr(args,'datasets',None) or c['retrieval']['datasets']:
        if not (data/'indexes'/ds/'manifest.json').exists(): raise Blocked('Complete index missing: '+ds)
        base_retriever=Retriever(c,ds,encoder); rows=list(read_rows(data/'public'/ds/f'{split}.jsonl'))[:limit]; private={r['qid']:r for r in read_rows(data/'private'/ds/f'{split}.jsonl')}
        def process_question(row):
            import copy
            retriever=copy.copy(base_retriever); retriever.calls=[]
            ex=InferenceExample(**row)
            for method in methods:
                checkpoint=data/'checkpoints'/('rebind' if method in ['frozen_old_read','frozen_query_replay'] else method)/str(getattr(args,'seed',17))/'best.pt'; neural=None
                if stage=='smoke' and not checkpoint.exists() and method in ['bp_rebind','rebind']: checkpoint=data/'smoke_initialization'/method/'initial.pt'
                if method in ['bp_rebind','rebind','independent_binding','no_revision_loss','frozen_old_read','frozen_query_replay']:
                    if not checkpoint.exists(): blocked.append(dict(dataset=ds,method=method,reason='checkpoint missing')); continue
                    neural=NeuralState(encoder,c,checkpoint,'rebind' if method=='frozen_query_replay' else method,split=split)
                file=phase_dir/(ds+'_'+row['qid']+'_'+method+('_seed'+str(args.seed) if getattr(args,'seed',17)!=17 and neural else '')+'.json')
                if file.exists() and args.resume:
                    cached=json.loads(file.read_text())
                    if cached.get('config_hash')==digest(c) and cached.get('pipeline_hash')==pipeline_hash and (neural is None or cached.get('checkpoint_hash')==digest(checkpoint)): results.append(cached); continue
                generator.context=dict(qid=ex.qid,split=split,protocol=protocol,candidate_proposer='shared-v1',source_hash=digest(root/'manifests/sources.lock.json'))
                start=time.time(); callstart=len(generator.calls); querystart=len(retriever.calls)
                try:
                    fixed=None
                    if protocol=='fixed' or method=='frozen_query_replay':
                        commonfile=root/'runs/natural_evaluate_open'/(ds+'_'+ex.qid+'_ircot_common.json')
                        if not commonfile.exists(): raise Blocked('Fixed IRCoT schedule missing: '+str(commonfile))
                        fixed=json.loads(commonfile.read_text())['trace']
                    if method=='ircot_native':
                        output=native_run(ex,retriever,generator,root); answer=output['pred']
                        pred=dict(qid=ex.qid,method=method,answer=answer,citations=[],eval_status='ok' if 'So the answer is:' in output['raw_pred'] else 'no_terminal_answer',fallback=0,output=output,trace=output.get('retrieval_trace',[]),query_count=len(retriever.calls)-querystart,llm_calls=len(generator.calls)-callstart,config_hash=digest(c))
                    elif method=='pyrag':
                        from .adapters.pyrag import run as pyrag_run
                        output=pyrag_run(ex,retriever,generator,root)
                        pred=dict(qid=ex.qid,method=method,answer=str(output.get('final_answer','')),citations=[],eval_status=output.get('eval_status','ok' if output.get('final_answer') else 'no_terminal_answer'),fallback=0,output=output,trace=output.get('retrieval_trace',[]),query_count=len(retriever.calls)-querystart,config_hash=digest(c))
                    else: pred=run_loop(ex,retriever,generator,c,method,neural,fixed)
                except Blocked as e: blocked.append(dict(dataset=ds,qid=ex.qid,method=method,reason=str(e))); continue
                except Exception as e:
                    import traceback
                    pred=dict(qid=ex.qid,method=method,answer='',citations=[],eval_status='runtime_failure',fallback=1,error=str(e),traceback=traceback.format_exc(),trace=[],config_hash=digest(c))
                pred.update(executed_retrieval_calls=len(retriever.calls)-querystart,replayed_retrieval_calls=len(pred.get('trace',[])) if protocol=='fixed' or method=='frozen_query_replay' else 0)
                actual_calls=generator.calls[callstart:]
                pred['model_response_keys']=[x['key'] for x in actual_calls]
                pred.update(uncached_llm_calls=sum(not x['cache_hit'] for x in actual_calls),uncached_input_tokens=sum(x['input_tokens'] for x in actual_calls if not x['cache_hit']),uncached_output_tokens=sum(x['output_tokens'] for x in actual_calls if not x['cache_hit']),api_calls_with_unknown_output_tokens=sum(x.get('output_tokens_unknown',False) for x in actual_calls))
                pred.update(llm_calls=len(actual_calls),llm_input_tokens=sum(x['input_tokens'] for x in actual_calls),llm_output_tokens=sum(x['output_tokens'] for x in actual_calls),cache_hits=sum(x['cache_hit'] for x in actual_calls))
                pred.update(training_scope=neural.metadata.get('scope') if neural else 'frozen_baseline',training_data_hash=neural.metadata.get('data_hash') if neural else None,parameter_count=sum(p.numel() for p in neural.model.parameters()) if neural else None,process_peak_module_gpu_bytes=peak_memory(neural.device) if neural else None)
                pred.update(training_status='untrained_smoke' if 'smoke_initialization' in str(checkpoint) else ('trained' if neural else 'frozen_baseline'))
                pred.update(pipeline_hash=pipeline_hash,dataset=ds,split=split,protocol=protocol,training_seed=getattr(args,'seed',17) if neural else None,checkpoint_hash=digest(checkpoint) if neural else None,seconds=time.time()-start)
                pred.update(official_scores(root,ds,pred['answer'],private[ex.qid]))
                support=set(private[ex.qid]['supporting_doc_ids']); trace=pred.get('trace',[])
                initial={d['parent_doc_id'] for d in trace[0]['retrieved_documents']} if trace else set(); final={d['parent_doc_id'] for t in trace for d in t['retrieved_documents']}
                pred.update(initial_gold_support_recall=len(initial&support)/len(support) if support and trace else None,final_gold_support_recall=len(final&support)/len(support) if support and trace else None,new_gold_support_count=len((final-initial)&support))
                if file.exists():
                    previous=json.loads(file.read_text()); history=phase_dir/'history'/previous.get('pipeline_hash','initial')/file.name; history.parent.mkdir(parents=True,exist_ok=True); file.replace(history)
                write(file,pred); append(phase_dir/'predictions.jsonl',pred); results.append(pred)
                print(stage,ds,ex.qid,method,pred['eval_status'],'EM',pred['em'],'seconds',round(pred['seconds'],2),flush=True)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS',c.get('execution',{}).get('workers',8)))) as pool: list(pool.map(process_question,rows))
    summary=[]
    for ds in getattr(args,'datasets',None) or c['retrieval']['datasets']:
        for method in methods:
            rr=[r for r in results if r['dataset']==ds and r['method']==method]
            if rr: summary.append(dict(dataset=ds,method=method,n=len(rr),em=float(np.mean([r['em'] for r in rr])),f1=float(np.mean([r['f1'] for r in rr])),failures=sum(r['eval_status']!='ok' for r in rr)))
    write(phase_dir/'summary.json',dict(results=summary,blocked=blocked)); return dict(results=summary,blocked=blocked)

def report(c):
    """Rebuild tables from real per-question files, never from a stage's success flag."""
    import math,re
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); reports=root/'reports'; groups={}; gold={}; questions={}
    transition_manifest=data/'natural_transitions/manifest.json'
    if transition_manifest.exists():
        trained_summaries=[json.loads(p.read_text()) for p in reports.glob('natural_training_seed*.json')]
        trained_summaries=[r for r in trained_summaries if r.get('data_hash')==digest(transition_manifest)]
        if trained_summaries:
            combined=dict(trained_summaries[0],results=[r for summary in trained_summaries for r in summary['results']]);write(reports/'natural_training.json',combined)
            expected={(m,seed) for m in ['bp_rebind','rebind','independent_binding','no_revision_loss'] for seed in c['train']['seeds']}
            completed={(r['method'],r['seed']) for r in combined['results']}
            if completed==expected:
                checkpoints=[p for p in (data/'checkpoints').glob('*/*/*.pt')]
                write(root/'manifests/train.json',dict(status='complete',aggregation='Actual independently executed train_<seed> stages; no additional training implied',result=combined,source_manifests=['train_'+str(seed)+'.json' for seed in c['train']['seeds']],config_sha256=digest(c),output_hashes={str(p.relative_to(root)):digest(p) for p in checkpoints+[reports/'natural_training.json']}))
    for ds in c['retrieval']['datasets']:
        for split in ['dev','eval']:
            questions.update({(ds,r['qid']):r['question'] for r in read_rows(data/'public'/ds/f'{split}.jsonl')})
            for r in read_rows(data/'private'/ds/f'{split}.jsonl'):
                gold[(ds,r['qid'])]=r
                groups[(ds,r['qid'])]=digest(r.get('evidences') or r.get('gold_decomposition') or r['qid'])
    records=[]; failures=[]; compact=[]
    for folder in sorted((root/'runs').glob('natural_*')):
        for path in sorted(folder.glob('*.json')):
            r=json.loads(path.read_text())
            if 'qid' not in r or 'em' not in r: continue
            ds=r['dataset']; qid=r['qid']; trace=r.get('trace',[]); label=gold[(ds,qid)]; support=set(label['supporting_doc_ids']); spans=collections.defaultdict(list); parents={}; seen=set(); retrieved=set(); citation_parent={}
            for step in trace:
                for d in step['retrieved_documents']:
                    parents[d['doc_id']]=d['parent_doc_id']; retrieved.add(d['parent_doc_id']); citation_parent[d['doc_id']]=d['parent_doc_id']
                seen.update(step.get('seen_ids',[]))
                for item in step.get('visible_spans',[]):
                    if item['doc_id'] in parents: spans[parents[item['doc_id']]].append(item['offsets'])
            if not trace: r.update(initial_gold_support_recall=None,final_gold_support_recall=None,new_gold_support_count=None)
            readparents={parents[x] for x in seen if x in parents}; facts=label['supporting_facts']
            validfacts=[f for f in facts if isinstance(f,dict) and 'offsets' in f]
            sentences_read=sum(any(a<=f['offsets'][0] and b>=f['offsets'][1] for a,b in spans[f['doc_id']]) for f in validfacts)
            citations=r.get('citations',[]); r.update(phase=folder.name,artifact=str(path.relative_to(root)),seed=r.get('training_seed') or 'frozen',retrieved_document_count=len(retrieved) if trace else None,seen_document_count=len(readparents) if trace else None,actual_read_gold_support_recall=len(readparents&support)/len(support) if trace and support else None,all_support_retrieved=float(support<=retrieved) if trace and support else None,sentence_read_recall=sentences_read/len(validfacts) if validfacts and trace else None,citation_gold_overlap=sum(citation_parent.get(x) in support for x in citations)/len(citations) if citations else None,citation_locatable_rate=r.get('citation_locatable',sum(x in parents for x in citations))/len(citations) if citations else None)
            structural_steps=[t for t in trace if 'proposal' in t]
            r['support_mapping_available']=bool(support)
            if structural_steps:
                accepted=sum(len(t['proposal']['accepted']) for t in structural_steps);rejected=sum(len(t['proposal']['rejected']) for t in structural_steps)
                values=[value for t in structural_steps for assignment in t.get('state',{}).get('assignments',[])[:1] if isinstance(assignment,dict) for value in assignment.values()]
                r.update(proposal_accepted=accepted,proposal_rejected=rejected,proposal_acceptance_rate=accepted/(accepted+rejected) if accepted+rejected else None,decoded_unknown_fraction=sum(value=='UNKNOWN' for value in values)/len(values) if values else None)
                r['nominal_triangle_step_fraction']=sum(len(t.get('candidates',{}))>=3 and any(len(domain)>1 for domain in t.get('candidates',{}).values()) for t in structural_steps)/len(structural_steps)
            if r.get('checkpoint_hash'):
                changes=[change for t in trace for change in t.get('state',{}).get('revisions',[]) if isinstance(change,dict) and change.get('verified') is False]
                r.update(unknown_to_candidate_events=sum(x['before']=='UNKNOWN' and x['after']!='UNKNOWN' for x in changes),candidate_to_unknown_events=sum(x['before']!='UNKNOWN' and x['after']=='UNKNOWN' for x in changes),unverified_candidate_switches=sum(x['before']!='UNKNOWN' and x['after']!='UNKNOWN' for x in changes),eligible_verified_revision_count=0)
            records.append(r)
            compact.append({k:r.get(k) for k in ['qid','dataset','split','phase','protocol','method','training_seed','checkpoint_hash','config_hash','pipeline_hash','answer','citations','em','f1','eval_status','fallback','query_count','executed_retrieval_calls','replayed_retrieval_calls','llm_calls','llm_input_tokens','llm_output_tokens','cache_hits','uncached_llm_calls','uncached_input_tokens','uncached_output_tokens','api_calls_with_unknown_output_tokens','seconds','artifact']})
            if r['eval_status']!='ok' or r.get('fallback',0): failures.append(dict(qid=qid,dataset=ds,method=r['method'],phase=folder.name,status=r['eval_status'],fallback=r.get('fallback',0),error=r.get('error'),artifact=r['artifact']))
    def csvfile(name,rows):
        if not rows: rows=[dict(status='not_run')]
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (reports/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    def summaries(rows):
        out=[]; keys=sorted({(r['phase'],r['dataset'],r['method'],str(r['seed'])) for r in rows})
        metrics=['em','f1','query_count','executed_retrieval_calls','replayed_retrieval_calls','llm_calls','llm_input_tokens','llm_output_tokens','cache_hits','uncached_llm_calls','uncached_input_tokens','uncached_output_tokens','api_calls_with_unknown_output_tokens','seconds','initial_gold_support_recall','final_gold_support_recall','new_gold_support_count','actual_read_gold_support_recall','all_support_retrieved','sentence_read_recall','citation_locatable_rate','citation_gold_overlap','retrieved_document_count','seen_document_count','proposal_accepted','proposal_rejected','proposal_acceptance_rate','decoded_unknown_fraction','nominal_triangle_step_fraction','unknown_to_candidate_events','candidate_to_unknown_events','unverified_candidate_switches','parameter_count','process_peak_module_gpu_bytes']
        for phase,ds,method,seed in keys:
            rr=[r for r in rows if (r['phase'],r['dataset'],r['method'],str(r['seed']))==(phase,ds,method,seed)]; requested=500 if rr[0]['split']=='eval' else c.get('execution',{}).get('smoke_questions',16) if phase.startswith('natural_smoke_') else 200
            entry=dict(phase=phase,dataset=ds,method=method,seed=seed,n=len(rr),requested=requested,missing=requested-len(rr),status='complete_sample' if len(rr)==requested else 'partial_sample',failures=sum(r['eval_status']!='ok' for r in rr),fallback_questions=sum(bool(r.get('fallback')) for r in rr),pipeline_versions=';'.join(sorted({r.get('pipeline_hash','legacy')[:12] for r in rr})))
            mapped=[r for r in rr if r['support_mapping_available']];entry.update(support_mapped_n=len(mapped),support_mapped_em=float(np.mean([r['em'] for r in mapped])) if mapped else None,support_mapped_f1=float(np.mean([r['f1'] for r in mapped])) if mapped else None)
            for metric in metrics:
                values=[r[metric] for r in rr if r.get(metric) is not None]; entry[metric]=float(np.mean(values)) if values else None
            out.append(entry)
        return out
    internal={'ircot_native','ircot_common','json_rebind','bp_rebind','rebind'}; ablations={'frozen_old_read','independent_binding','no_revision_loss','frozen_query_replay'}
    fixed_checks=[]
    for ds in c['retrieval']['datasets']:
        by_question=collections.defaultdict(dict)
        for r in records:
            if r['phase']=='natural_evaluate_fixed' and r['dataset']==ds and r['seed'] in ['frozen',17]: by_question[r['qid']][r['method']]=r
        for qid,methods in by_question.items():
            if not {'direct_reader','json_rebind','bp_rebind','rebind'}<=methods.keys(): continue
            raw_signatures={digest([(t['query'],t['retrieved_ids'],t.get('visible_spans',[])) for t in methods[m]['trace']]) for m in ['direct_reader','json_rebind','bp_rebind','rebind']}
            candidate_signatures={digest([t.get('candidates') for t in methods[m]['trace']]) for m in ['json_rebind','bp_rebind','rebind']}
            fixed_checks.append(dict(dataset=ds,qid=qid,same_raw_schedule=len(raw_signatures)==1,same_candidate_schedule=len(candidate_signatures)==1,statuses={m:r['eval_status'] for m,r in methods.items()},errors={m:[t['error'] for t in r['trace'] if 'error' in t] for m,r in methods.items() if any('error' in t for t in r['trace'])}))
            b,r=methods['bp_rebind'],methods['rebind']
            fixed_checks[-1].update(bp_rebind_same_top_assignment_schedule=[t.get('state',{}).get('assignments',[])[:1] for t in b['trace']]==[t.get('state',{}).get('assignments',[])[:1] for t in r['trace']],bp_rebind_same_answer_string=b['answer']==r['answer'],rebind_minus_bp_em=r['em']-b['em'])
    write(reports/'fixed_protocol_audit.json',dict(records=fixed_checks,total=len(fixed_checks),raw_matched=sum(r['same_raw_schedule'] for r in fixed_checks),candidate_matched=sum(r['same_candidate_schedule'] for r in fixed_checks),interpretation='Actual trace comparison; failures remain in full QA denominators. Candidate matching compares structured methods only.'))
    natural=summaries([r for r in records if r['phase']=='natural_evaluate_open' and r['method'] in internal]); fixed=summaries([r for r in records if r['phase']=='natural_evaluate_fixed']); ablation=summaries([r for r in records if r['phase']=='natural_evaluate_open' and r['method'] in ablations]); external=summaries([r for r in records if r['method'] in ['pyrag','reagent']]); diagnostic=summaries([r for r in records if r['split']=='dev' and r['method'] in internal|ablations])
    external.append(dict(method='reagent',status='blocked',n=0,reason='Official Agent/agent.py references undefined Agent after explicit backend import retry',evidence='runs/initial_20260914/reagent_explicit_import_retry.log'))
    cp=reports/'controlled_reader_results.json'
    if cp.exists():
        cr=json.loads(cp.read_text())
        for method in sorted({r['method'] for r in cr}):
            rr=[r for r in cr if r['method']==method];diagnostic.append(dict(phase='controlled_fixed_4_worlds',dataset='program_worlds',method=method,n=len(rr),em=float(np.mean([r['em'] for r in rr])),status='controlled_only',evidence=cp.name))
    bindingpath=reports/'natural_binding_diagnostic.json'
    if bindingpath.exists():
        for row in json.loads(bindingpath.read_text())['results']:
            diagnostic.append(dict(phase='weak_natural_prefix_binding_dev',dataset='2wiki',status='partial_automatic_labels',evidence=bindingpath.name,**row))
    for name,rows in [('results_natural.csv',natural),('results_fixed_evidence.csv',fixed),('results_ablations.csv',ablation),('results_external.csv',external),('results_diagnostic.csv',diagnostic)]: csvfile(name,rows)
    # Paired group bootstrap: sample complete base groups, not individual prefixes or duplicated seeds.
    comparisons=[]
    for phase in ['natural_evaluate_open','natural_evaluate_fixed']:
        for ds in c['retrieval']['datasets']:
            for seed in c['train']['seeds']:
                treatment={r['qid']:r for r in records if r['phase']==phase and r['dataset']==ds and r['method']=='rebind' and r['seed']==seed}
                for control in ['ircot_common','direct_reader','json_rebind','bp_rebind']:
                    baseline={r['qid']:r for r in records if r['phase']==phase and r['dataset']==ds and r['method']==control and r['seed'] in ['frozen',seed]}; ids=sorted(treatment.keys()&baseline.keys())
                    if not ids: continue
                    grouped=collections.defaultdict(list)
                    for qid in ids: grouped[groups[(ds,qid)]].append(qid)
                    gs=list(grouped.values()); sizes=np.array([len(g) for g in gs]); rng=np.random.default_rng(c['evaluation']['bootstrap_seed']); draw=rng.integers(0,len(gs),(c['evaluation']['bootstrap_samples'],len(gs)))
                    for metric in ['em','f1']:
                        sums=np.array([sum(treatment[q][metric]-baseline[q][metric] for q in group) for group in gs]); samples=sums[draw].sum(1)/sizes[draw].sum(1); lo,hi=np.quantile(samples,[.025,.975])
                        comparisons.append(dict(phase=phase,dataset=ds,seed=seed,treatment='rebind',control=control,metric=metric,n=len(ids),groups=len(gs),difference=float(sums.sum()/sizes.sum()),ci_low=float(lo),ci_high=float(hi),resamples=c['evaluation']['bootstrap_samples']))
    csvfile('paired_bootstrap.csv',comparisons)
    if comparisons:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
        for row,phase in enumerate(['natural_evaluate_fixed','natural_evaluate_open']):
            for column,ds in enumerate(c['retrieval']['datasets']):
                ax=axes[row,column];rr=[r for r in comparisons if r['phase']==phase and r['dataset']==ds and r['metric']=='em' and r['seed']==17 and r['n']==500]
                ax.axvline(0,color='#777777',linewidth=1)
                for y,r in enumerate(rr):
                    value=100*r['difference'];ax.errorbar(value,y,xerr=[[value-100*r['ci_low']],[100*r['ci_high']-value]],fmt='o',color='#275D8C',capsize=4)
                ax.set(yticks=range(len(rr)),yticklabels=[r['control'] for r in rr],xlabel='ReBind minus control EM (percentage points)',title=('Fixed evidence' if row==0 else 'Open retrieval')+' | '+ds)
                ax.grid(axis='x',alpha=.2)
                if not rr:ax.text(.5,.5,'Full paired sample pending',ha='center',transform=ax.transAxes)
        fig.suptitle('Seed 17 | 500 questions per dataset | 95% paired group bootstrap CI');fig.savefig(reports/'paired_effects.png',dpi=220);plt.close(fig)
    matched={(r['dataset'],r['qid']) for r in fixed_checks if r['same_raw_schedule'] and r['same_candidate_schedule']}
    sensitivity=summaries([r for r in records if r['phase']=='natural_evaluate_fixed' and (r['dataset'],r['qid']) in matched])
    for r in sensitivity:
        r.update(status='posthoc_matched_input_sensitivity',excluded_from_full_sample=r.pop('missing'))
    csvfile('fixed_matched_sensitivity.csv',sensitivity)
    seedstats=[]
    for phase in ['natural_evaluate_open','natural_evaluate_fixed']:
        for ds in c['retrieval']['datasets']:
            for method in ['rebind','bp_rebind','independent_binding','no_revision_loss']:
                rr=[r for r in natural+fixed+ablation if r['phase']==phase and r['dataset']==ds and r['method']==method]
                if rr: seedstats.append(dict(phase=phase,dataset=ds,method=method,seeds=len(rr),em_mean=float(np.mean([r['em'] for r in rr])),em_std=float(np.std([r['em'] for r in rr],ddof=1)) if len(rr)>1 else None,f1_mean=float(np.mean([r['f1'] for r in rr])),f1_std=float(np.std([r['f1'] for r in rr],ddof=1)) if len(rr)>1 else None))
    csvfile('training_seed_summary.csv',seedstats)
    curves=[]; path=reports/'natural_learning_curves.jsonl'
    if path.exists(): curves=list(read_rows(path))
    csvfile('learning_curves.csv',curves)
    if curves:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for column,seed in enumerate(c['train']['seeds']):
            for row,field in enumerate(['train_loss','dev_loss']):
                ax=axes[row,column]
                for method in sorted({r['method'] for r in curves}):
                    rr=[r for r in curves if r['method']==method and r['seed']==seed and r.get('data_hash')==curves[-1].get('data_hash')]
                    ax.plot([r['epoch']+1 for r in rr],[r[field] for r in rr],marker='x' if method=='no_revision_loss' else 'o',linestyle='--' if method=='no_revision_loss' else '-',label=method)
                ax.set(xlabel='Epoch',ylabel='Weak binding loss (log)',yscale='log',xticks=[1,2,3],title=field.replace('_',' ').title()+f' | seed {seed}');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.savefig(reports/'learning_curves.png',dpi=220);plt.close(fig)
    with (root/'runs/initial_20260914/predictions.jsonl').open('w') as f:
        for r in compact: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    prior=list(read_rows(reports/'failure_cases.jsonl')) if (reports/'failure_cases.jsonl').exists() else []; existing={digest(r) for r in prior}
    for r in failures:
        if digest(r) not in existing: append(reports/'failure_cases.jsonl',r)
    phases=['preflight','audit-sources','prepare-data','build-index','audit-data','audit-retrieval','diagnose','prepare-transitions','smoke','train','evaluate_fixed','evaluate_open','intervene']; completion={}
    for stage in phases:
        path=root/'manifests'/(stage+'.json'); completion[stage]=json.loads(path.read_text()) if path.exists() else dict(status='not_run')
    for phase,table in [('evaluate_open',natural),('evaluate_fixed',fixed)]:
        needed=internal if phase=='evaluate_open' else {'direct_reader','json_rebind','bp_rebind','rebind'}
        counts={(r['dataset'],r['method']):r['n'] for r in table if r['seed'] in ['17','frozen']}
        complete=all(counts.get((ds,m),0)==500 for ds in c['retrieval']['datasets'] for m in needed)
        completion[phase].update(status='complete_sample' if complete else 'partial' if table else 'not_run',observed_counts={ds+':'+m:n for (ds,m),n in counts.items()})
    diagnose_counts={(ds,m):sum(r['phase']=='natural_diagnose_open' and r['dataset']==ds and r['method']==m for r in records) for ds in c['retrieval']['datasets'] for m in ['ircot_native','ircot_common','json_rebind']}
    completion['diagnose']['status']='complete_sample' if all(n==200 for n in diagnose_counts.values()) else 'partial' if any(diagnose_counts.values()) else 'not_run'
    if completion['prepare-transitions'].get('arguments',{}).get('limit'): completion['prepare-transitions']['status']='partial_pilot'
    trained=list((data/'checkpoints').glob('*/*/best.pt'))
    completion['train']['status']='complete_training_runs' if len(trained)==12 and not completion['prepare-transitions'].get('arguments',{}).get('limit') else 'partial' if trained else 'not_run'
    completion['scientific_scope']=dict(status='preliminary',natural_revision_supervision='unavailable',source_scope_supervision='controlled_only',external_reagent='blocked',causal_H2_H3='not_established',natural_training_order='controlled presentation of natural training documents; open-trajectory training not run',mixed_revision_training='not_run; controlled diagnostics are separate',constraint_scope='approximate beam; type, identity, role and date checks apply only where verified metadata exists; no claim of complete natural constraint coverage')
    write(reports/'completion_manifest.json',completion)
    labelpath=reports/'natural_label_audit.json'; label=json.loads(labelpath.read_text()) if labelpath.exists() else dict(status='not_run')
    oldlabel=json.loads((reports/'label_audit.json').read_text()) if (reports/'label_audit.json').exists() else {}
    controlled=oldlabel.get('controlled',{k:v for k,v in oldlabel.items() if k!='natural'});controlled.pop('training_natural_transitions',None);controlled['status']='controlled_disambiguation_and_confirmation_only'
    write(reports/'label_audit.json',dict(status='completed_with_partial_natural_supervision',controlled=controlled,natural=label))
    lines=['# ReBind-RAG 实际运行报告','', '本报告由逐题结果重新汇总。当前研究状态为 preliminary：自然修订/来源作用域标签不足，ReAgent 官方代码适配仍阻塞。程序微型训练、自然弱监督训练、同文理解和开放检索分别报告。','', '## 已执行范围','', '| 阶段 | 状态 | 证据 |','|---|---|---|']
    errorpath=reports/'natural_error_diagnosis.json'
    if errorpath.exists():
        errorcounts=json.loads(errorpath.read_text())['counts']
        lines[4:4]=['自然绑定错误覆盖尚未得到可靠总体估计。400 题开发诊断的自动类别（允许重叠）为：'+json.dumps(errorcounts,ensure_ascii=False)+'。标注支持未召回不排除替代正确证据；答案可见而答错不能自动判定为绑定错误。固定顺序的六个案例仅作 model_judged 复核。','']
    for stage in phases: lines.append(f"| {stage} | {completion[stage]['status']} | [manifest](../manifests/{stage}.json) |")
    lines+=['','所有阶段的实际命令、退出码和时间见 [commands.jsonl](../runs/initial_20260914/commands.jsonl)。逐题索引见 [predictions.jsonl](../runs/initial_20260914/predictions.jsonl)，其中 artifact 指向原始轨迹。失败进入相应分母；未执行题报告为 missing，不填充假成绩。','', '## 自然总体与同文结果','']
    for title,table,name in [('自然开放检索',natural,'results_natural.csv'),('同文证据',fixed,'results_fixed_evidence.csv')]:
        lines += [f'### {title}',f'[完整表](./{name})','', '| 数据集 | 方法 | seed | n | EM | F1 | 非正常终态 | 发生回退的题数 |','|---|---|---|---:|---:|---:|---:|---:|']
        for r in table: lines.append(f"| {r['dataset']} | {r['method']} | {r['seed']} | {r['n']} | {r['em']:.4f} | {r['f1']:.4f} | {r['failures']} | {r['fallback_questions']} |")
        if not table: lines.append('| — | not_run | — | — | — | — | — | — |')
    lines+=['','## 研究判断','', 'H1（相同原文下，联合绑定比强 JSON/BP 更准确）：同文 QA 与配对区间只提供任务层代理证据；最终 reader 能在绑定错误时答对，不能将 QA 增益直接当作绑定准确率提升。部分弱标签开发诊断单独报告，当前缺少可靠自然联合绑定标签，无法直接确认 H1。H2（绑定修订改变查询并找到初始遗漏来源）：开放检索和 fork 记录 query、旧来源分数与实际阅读差异；但缺少可信自然修订标签时，不能把任意分数变化认定为正确修订。H3（H2 新来源改善最终回答）：单独比较 fork 答案和召回；没有 H2 中间环节证据时，不能由 QA 分差直接宣布完整机制链成立。','', '[配对 bootstrap](./paired_bootstrap.csv) 使用基础事实组采样 2,000 次；[训练 seed 统计](./training_seed_summary.csv) 单列。生成器固定 greedy，基线只运行一次，不复制成三个 seed。','', '## 诊断、消融、外部方法','', '[诊断表](./results_diagnostic.csv) · [消融表](./results_ablations.csv) · [外部表](./results_external.csv) · [干预](./interventions_summary.json)','', 'NO_REVISION_LOSS 在无 eligible revision 标签的 natural-only 训练中是同构同损失对照，不能解释为修订损失无用。FROZEN_OLD_READ 是同 checkpoint 的推理干预，存在分布偏移，未冒称独立重训。受控微型世界的 train/dev 共享模板，不代表自然泛化或模板外泛化。','', '## 标签与训练','', '自然训练只使用当前前缀可对齐的弱监督，不把未来支持材料或 gold 图送入 forward。自然标签覆盖和定位证据见 [natural_label_audit.json](./natural_label_audit.json)。没有可信修订标签的项不构造伪纠错。','```json',json.dumps(label,ensure_ascii=False,indent=2),'```','', '[学习曲线 CSV](./learning_curves.csv)；保留 last/best 和开发损失选择记录。', '', '三个训练 seed 已分别运行；本轮封存 QA 使用预先固定的 seed 17，seed 29/43 的 QA 扩展未运行。开发侧整体下降、独立绑定诊断和固定顺序反例不支持继续扩大这套训练配方的 GPU 消耗，具体决策记录在封存输出前写入 decisions.jsonl。一个 QA seed 的标准差记 N/A。', '', '[开发集固定顺序案例复核](./development_case_review.md) 包含答案正确但绑定错误、有害状态变化与全部失败的例子；复核者为 Codex，标记 model_judged，不是 human_reviewed。']
    lines+=['','### 实际配对判断','']
    for r in comparisons:
        if r['metric']!='em' or r['seed']!=17: continue
        judgment='区间高于零' if r['ci_low']>0 else '区间低于零' if r['ci_high']<0 else '区间跨零，差异未获支持'
        lines.append(f"- {r['phase']} / {r['dataset']}，ReBind 对 {r['control']}：EM 差 {r['difference']:+.4f}，95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]，{judgment}，配对 n={r['n']}。")
    if not comparisons: lines.append('封存集配对结果尚未完成，不做收益判断。')
    if comparisons: lines+=['','![同文与开放检索的配对 EM 差异](./paired_effects.png)']
    lines+=['',f"同文实际输入核对：{len(fixed_checks)} 题中原文日程一致 {sum(r['same_raw_schedule'] for r in fixed_checks)} 题，结构候选日程一致 {sum(r['same_candidate_schedule'] for r in fixed_checks)} 题。主表保留所有失败；[事后匹配输入敏感性表](./fixed_matched_sensitivity.csv) 另列严格匹配子集，不替代预定全样本结果。候选差异及运行错误见 [逐题审计](./fixed_protocol_audit.json)。"]
    for ds in c['retrieval']['datasets']:
        rr=[r for r in fixed_checks if r['dataset']==ds]
        lines.append(f"同文 {ds}：BP 与 REBIND 的第一联合赋值日程相同 {sum(r['bp_rebind_same_top_assignment_schedule'] for r in rr)}/{len(rr)} 题；答案字符串相同 {sum(r['bp_rebind_same_answer_string'] for r in rr)}/{len(rr)} 题；REBIND 的 EM 改善 {sum(r['rebind_minus_bp_em']>0 for r in rr)} 题、变差 {sum(r['rebind_minus_bp_em']<0 for r in rr)} 题。状态变化不是自动核验的正确修订。")
    flips=[r for r in fixed_checks if r['rebind_minus_bp_em']]
    if flips:
        lines+=['','以下列出同文 BP/REBIND 的全部 EM 翻转，属于封存后描述性审计，没有用来改模型：','', '| 数据集 / qid | 问题 | BP 答案 / EM | REBIND 答案 / EM |','|---|---|---|---|']
        for check in flips:
            pair={r['method']:r for r in records if r['phase']=='natural_evaluate_fixed' and r['dataset']==check['dataset'] and r['qid']==check['qid'] and r['method'] in ['bp_rebind','rebind']};b,r=pair['bp_rebind'],pair['rebind'];question=questions[(check['dataset'],check['qid'])].replace('|',' / ')
            lines.append(f"| {check['dataset']} / {check['qid']} | {question} | [{b['answer'].replace('|',' / ')}](../{b['artifact']}) / {b['em']:.0f} | [{r['answer'].replace('|',' / ')}](../{r['artifact']}) / {r['em']:.0f} |")
    lines+=['','开发侧的 246 个有标签前缀仅有 330 个自动对齐变量标签。结构准确率见 [natural_binding_diagnostic.json](./natural_binding_diagnostic.json)，不能将其当作人工核验的完整联合绑定准确率。训练使用自然原文的受控到达顺序；自然开放轨迹训练与 natural+controlled 混合训练未执行。受控诊断单独列出，不能补足这一缺口。','', '切分额外检查见 [overlap_audit.json](./overlap_audit.json)：字符相似度阈值 0.92 下无跨切分近似问句；有 22 组共享精确子问题，未声称完全隔离模板、实体或子问题。']
    forkpath=reports/'interventions_summary.json'
    if forkpath.exists():
        forks=json.loads(forkpath.read_text()).get('pairs',[]); eligible=[r for r in forks if r['shared_prefix_and_new_evidence']]
        if forks:
            aggregate=dict(total=len(forks),shared_prefix_pairs=len(eligible),query_diverged=sum(r['queries_diverged'] for r in eligible),normal_more_new_support=sum(r['normal_new_gold_support_read']>r['frozen_new_gold_support_read'] for r in eligible),normal_em_better=sum(r['normal_em']>r['frozen_em'] for r in eligible),normal_em_worse=sum(r['normal_em']<r['frozen_em'] for r in eligible),normal_em_same=sum(r['normal_em']==r['frozen_em'] for r in eligible),verified_revision_denominator=None)
            f=json.loads(forkpath.read_text());f['aggregate']=aggregate;write(forkpath,f)
            lines+=['','### 实际 fork 结果','',json.dumps(aggregate,ensure_ascii=False),'','上述分母只包括共享前缀与新证据相同的配对。正确修订的自然标签缺失，因此 eligible revision 成功率为 N/A；查询或答案变化不能单独证明 H2/H3 的完整机制链。']
    if curves: lines+=['','![自然弱监督学习曲线](./learning_curves.png)']
    lines+=['','## 口径与限制','', '- 主检索为完整 IRCoT 配方语料的冻结 E5 exact IP；2Wiki 使用更新版数据，段落数与旧 IRCoT 版本不同。句子映射错误掩蔽并保留文档级结果。','- Gold-support recall 不判定其他来源错误；引用可定位率和支持重叠不是语义蕴含准确率。无句级标签/引用时报告 N/A。','- 原生 IRCoT 保留上游两轮和句号停止规则；COMMON 的预算与终答规则变化单列，不能将二者差异全归因于重绑定。','- 所有结构方法共享问题图与原文候选函数。开放检索轨迹不同会产生不同候选；只有固定证据实验可作为同候选对照。','- 调用及 tokens 同时报逻辑开销与未命中缓存的实际请求；墙钟延迟受缓存顺序影响，不作冷启动速度比较。HTTP 500 未返回 token 用量的请求单列，输出 token 总数为下界。','- source-local ID 不自动合并同名实体；beam 是近似搜索，未实现的硬逻辑约束不作已验证声明。','- 两变量图没有不同于两端的第三个桥接变量，无法提供该三角算子的非平凡消息。表中 nominal_triangle_step_fraction 仅统计变量数至少三且存在非 UNKNOWN 候选的步骤，不代表有可核验三角监督。', '- 自然 frontier 标签复用当前已对齐的变量允许集合，是弱代理监督，未验证真实的下一跳查询参数目标。', '- 同文 direct_reader 的控制循环执行了未用于最终答案的中间 reasoning 调用；表中如实计入这些开销，它不是经过最小调用优化的直接 reader。不得据此宣称计算效率优势。', '- source_scores 是窗口级数值；正确来源角色/作用域变更缺少可靠自然标签，其准确率为 N/A。未核验的候选切换与 UNKNOWN 变确定单列，不能叫正确纠错。', '- 学习信号主要是部分 unary 绑定；有效 pair/来源作用域监督缺口直接限制架构结论。继续扩跑必须先看自然 QA 与强对照差异，不因合成训练损失下降而扩占 GPU。','', '## 失败、调整与复现','', '[全部运行模型开销账本](./model_usage_ledger.json) · [同文输入核对](./fixed_protocol_audit.json) · [调整记录](../runs/initial_20260914/decisions.jsonl) · [失败样例](./failure_cases.jsonl) · [自然错误诊断](./natural_error_diagnosis.json) · [来源锁定](../manifests/sources.lock.json) · [兼容性](./baseline_compatibility.md)','', '```bash','cd .','.venv/bin/python -m rebind_mvp.cli report --config configs/resolved.yaml','```']
    fixed_em=[r for r in comparisons if r['phase']=='natural_evaluate_fixed' and r['metric']=='em' and r['seed']==17 and r['control'] in ['json_rebind','bp_rebind']]
    fixed_text='；'.join(f"{r['dataset']} 对 {r['control']}：{100*r['difference']:+.1f} 个百分点，95% CI [{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}]" for r in fixed_em) or '尚无已运行的配对结果'
    open_em=[r for r in comparisons if r['phase']=='natural_evaluate_open' and r['metric']=='em' and r['seed']==17 and r['control']=='ircot_common']
    open_text='；'.join(f"{r['dataset']} 对 COMMON：{100*r['difference']:+.1f} 个百分点（n={r['n']}，95% CI [{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}]）" for r in open_em) or '开放配对结果尚未产生'
    lines[2:2]=['## 六项直接回答','',
        f"1. **执行范围**：12 个自然训练运行、数值/集成测试、smoke、受控诊断和自然 fork 已实际执行；同文状态 `{completion['evaluate_fixed']['status']}`，开放状态 `{completion['evaluate_open']['status']}`。各阶段命令与证据见下表。ReAgent blocked；自然开放轨迹训练、混合修订训练和额外两个 seed 的 QA 未运行；没有人工审核完成声明。",
        '2. **自然错误与标签**：尚不能证明自然任务中有足够可信的角色/作用域修订样本。6,379 个可用前缀仅得到 3,291 unary、72 pair 和 3,291 frontier 代理标签；自然 revision/source-scope 标签均为零。所有自动对齐标签未经过人工核验。',
        '3. **同文 H1**：'+fixed_text+'。当前没有优于 JSON/BP 的证据，且缺少直接的可靠联合绑定评价。四个受控开发世界中所有 reader 都为 4/4，合成结果同样没有独特优势。JSON 在同文两数据集分别有 147/500、233/500 题发生回退，常见原因是嵌套赋值不符合字符串接口；因此不能称为已充分验证的强 JSON 上限。',
        '4. **开放 H2/H3**：'+open_text+'。32 个预定 fork 中只有 1 个查询分叉，0 个新增金标支持优势、0 个答案改善、1 个答案变差。这个样本没有支持预期机制链；自然正确修订标签缺失也限制了判断。',
        '5. **收益归因**：公共 parent/chunk ID 修复使候选可用性改善，但没有独立因子实验能量化接口、监督、计算、候选覆盖各自的因果贡献。学习方法使用额外弱监督；JSON 使用更多状态生成调用；主表给出实际调用和候选统计。缓存影响耗时，不能据此声称速度优势。封存后只做报告审计，没有针对这些答案改模型或接口。',
        '6. **近邻与下一步**：[近邻审计](./related_work_audit.md) 显示显式绑定、联合推断、回溯和图更新均已有先例。本实现的来源重读与同候选三角消息是待验证的组合，不足以支持完整增量结构监督的新颖性主张。当前最值得先做的是独立于本封存集的接口修复与回归检查，以及小规模人工核验的角色/作用域修订数据；随后在新的未见保留集检验同文绑定准确率与 fork 机制链。现有证据不支持继续扩大同一弱监督训练配方。','']
    (reports/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');return completion
