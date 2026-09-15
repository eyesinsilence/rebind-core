"""Exploratory few-shot and same-evidence diagnostics; no training or test-driven selection."""
import argparse,collections,csv,inspect,json,os,pathlib,time,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .audit import digest,write,append
from .schema import InferenceExample
from .proposal import Generator
from .evaluate import run_loop,JSON_STATE_PROMPT
from .mquake import EditRetriever,official_function


class FewShotGenerator(Generator):
    def __init__(self,c,demonstrations):
        super().__init__(c);self.demonstrations=demonstrations
    def json(self,prompt,max_tokens=None,schema=None):
        if prompt.startswith(JSON_STATE_PROMPT):
            prompt=JSON_STATE_PROMPT+'The following are examples of revising an intermediate binding. They are demonstrations only, not evidence about the current question. Next-query effects illustrate the existing scheduler; return only the requested binding-state JSON.\n'+self.demonstrations+'\nNow process the current question using only its own candidates and documents.\n'+prompt[len(JSON_STATE_PROMPT):]
        return super().json(prompt,max_tokens=max_tokens,schema=schema)


def prepare(c):
    import pyarrow.parquet as pq
    root=pathlib.Path(c['paths']['workdir']);parent=pathlib.Path(c['diagnostics']['parent']);split=json.loads((root/'data/adapt/split.json').read_text());train=set(split['split']['train']);rows={r['case_id']:r for r in pq.read_table(root/'data/raw/T-00000-of-00001.parquet').to_pylist()};samples=json.loads((root/'data/adapt/samples.json').read_text());groups={r['case_id']:r['group'] for r in samples}
    selected=[424,1072,1342];assert set(selected)<=train and len({groups[i] for i in selected})==3
    demos=[]
    for case_id in selected:
        r=rows[case_id];i=next(i for i,(a,b) in enumerate(zip(r['orig_triples'],r['new_triples'])) if a!=b)
        old=r['single_hops'][i];new=r['new_single_hops'][i];old_fact=old['cloze']+' '+old['answer'];new_fact=new['cloze']+' '+new['answer'];assert old['answer']!=new['answer']
        before={'v0':old['answer'],'answer':'UNKNOWN'};after={'v0':new['answer'],'answer':'UNKNOWN'}
        demos.append(dict(case_id=case_id,group=groups[case_id],question=r['questions'][0],old_document={'id':'demo_old','text':old_fact},initial_binding=before,new_document={'id':'demo_update','text':new_fact},candidates={'v0':['UNKNOWN',old['answer'],new['answer']],'answer':['UNKNOWN']},revised_state={'assignments':[after],'revisions':[{'variable':'v0','before':old['answer'],'after':new['answer']}],'observed_slots':['s0'],'observed_evidence':{'s0':{'doc_id':'demo_update','quote':new_fact}},'answer_ready':False},old_query= r['single_hops'][i+1]['question'],corrected_query_effect=r['new_single_hops'][i+1]['question']))
    payload='\n'.join(json.dumps({k:v for k,v in d.items() if k not in ['case_id','group']},ensure_ascii=False,separators=(',',':')) for d in demos)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(c['models']['generator_path'],local_files_only=True)
    audit=dict(demonstrations=demos,text=payload,tokens=len(tokenizer.encode(payload)),source='Official train-only old/new single-hop facts; controlled demonstrations, not naturally observed agent corrections; no final downstream answer included.',excluded=[dict(case_id=1213,reason='Headquarters city changes to Vietnam: city/country type mismatch.'),dict(case_id=1844,reason='Same edit group as1342; avoid duplicate demonstration.')],selection='Three distinct train edit groups with changed intermediate binding and changed next query. Selected by source validity before new test outputs; no accuracy-based prompt tuning.',scope='Previously evaluated training holdout, exploratory follow-up, not a new untouched test set.')
    write(root/'manifests/demonstrations.json',audit)
    return audit


def run(c,smoke=False):
    from .retrieval import E5
    from .transitions import NeuralState
    root=pathlib.Path(c['paths']['workdir']);parent=pathlib.Path(c['diagnostics']['parent']);data=root/'data';scope='smoke' if smoke else 'test';folder=root/'runs'/('diagnostics_'+scope);folder.mkdir(exist_ok=True)
    demos=json.loads((root/'manifests/demonstrations.json').read_text());split=json.loads((data/'adapt/split.json').read_text());ids=set(split['split']['dev'][:4] if smoke else split['split']['test']);rows=[r for r in json.loads((data/'public/T.json').read_text()) if r['case_id'] in ids and (not smoke or r['variant']==0)]
    parentlock=json.loads((parent/'manifests/adapt_qa_lock.json').read_text());selections=parentlock['identity']['all_trained_selections'];specs=[dict(name='json_zero',mode='json_rebind'),dict(name='json_fewshot',mode='json_rebind')]+[dict(name=f'trained_rebind_s{s}',mode='rebind',seed=s,**selections[f'rebind_s{s}']) for s in [17,29,43]]
    references={}
    if not smoke:
        for row in rows:
            for method in ['ircot_common']+[f'adapted_rebind_s{s}' for s in [17,29,43]]:
                p=parent/'runs/adapt_qa'/(row['qid']+'_'+method+'.json');references[str(p)]=digest(p)
    identity=dict(config_hash=digest(c),demonstrations_hash=digest(root/'manifests/demonstrations.json'),public_hash=digest(data/'public/T.json'),split_hash=digest(data/'adapt/split.json'),memory_hash=digest(data/'memory/T.json'),vectors_hash=digest(data/'memory/T_e5.npy'),parent_lock_hash=digest(parent/'manifests/adapt_qa_lock.json'),references=references,checkpoint_hashes={s['name']:digest(pathlib.Path(s['path'])) for s in specs if 'path' in s},runner=digest(inspect.getsource(run)),wrapper=digest(inspect.getsource(FewShotGenerator)),sources={p.name:digest(p) for p in (root/'src/rebind_mvp').glob('*.py') if p.name not in ['fewshot.py','mquake.py','adapt.py','cli.py','stages.py','data.py','diagnostics.py','interventions.py']},scope=scope)
    h=digest(identity);lock=root/'manifests'/('diagnostics_'+scope+'_lock.json')
    if lock.exists():assert json.loads(lock.read_text())['identity']==identity
    else:write(lock,dict(identity=identity,hash=h,time=time.time()))
    generator=Generator(c);few=FewShotGenerator(c,demos['text']);encoder=E5(c['models']['retriever_path'],device='cuda:3');docs=json.loads((data/'memory/T.json').read_text());vectors=np.load(data/'memory/T_e5.npy')
    def task(row):
        ex=InferenceExample(row['qid'],row['question']);cache={};outputs=[]
        context=dict(qid=ex.qid,split='diagnostics_'+scope,source_hash=identity['memory_hash'])
        generator.context=context;few.context=context
        if smoke:
            commonfile=folder/(ex.qid+'_open_common_reference.json')
            if commonfile.exists():common=json.loads(commonfile.read_text());assert common['identity_hash']==h
            else:
                common=run_loop(ex,EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k']),generator,c,'ircot_common')
                common.update(case_id=row['case_id'],variant=row['variant'],protocol='open',method='common_reference',identity_hash=h,imported=False);write(commonfile,common)
            outputs.append(common)
        else:
            for method,name in [('ircot_common','common_reference')]+[(f'adapted_rebind_s{s}',f'trained_rebind_s{s}') for s in [17,29,43]]:
                p=parent/'runs/adapt_qa'/(ex.qid+'_'+method+'.json');r=json.loads(p.read_text());assert r['identity_hash']==digest(parentlock['identity'])
                r.update(protocol='open',method=name,identity_hash=h,imported=True,parent_artifact=str(p),parent_artifact_hash=references[str(p)])
                write(folder/(ex.qid+'_open_'+name+'.json'),r);outputs.append(r)
                if name=='common_reference':common=r
        for protocol in ['open','fixed']:
            for spec in (specs[:2] if protocol=='open' else [dict(name='raw_reader',mode='direct_reader')]+specs):
                name=spec['name'];path=folder/(ex.qid+'_'+protocol+'_'+name+'.json')
                # Rebuild the shared proposer cache when resuming fixed methods, using persisted snapshots.
                cachefile=folder/(ex.qid+'_proposals.pt')
                if protocol=='fixed' and not cache and cachefile.exists():
                    import torch
                    cache.update(torch.load(cachefile,weights_only=False,map_location='cpu'))
                if path.exists():
                    r=json.loads(path.read_text());assert r['identity_hash']==h;outputs.append(r);continue
                g=few if name=='json_fewshot' else generator;g.context=context;first=len(g.calls);start=time.time();neural=None
                try:
                    if name=='raw_reader':
                        final=g.json(common['final_prompt']);assert isinstance(final['answer'],str)
                        r=dict(answer=final['answer'],trace=common['trace'],final_prompt=common['final_prompt'],fallback=0,eval_status='ok',state_included=False,raw_reader_also_no_state_lesion=True)
                    else:
                        if 'path' in spec:neural=NeuralState(encoder,c,pathlib.Path(spec['path']),'rebind',split='diagnostics_'+scope)
                        r=run_loop(ex,EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k']),g,c,spec['mode'],neural,fixed=common['trace'] if protocol=='fixed' else None,proposal_cache=cache if protocol=='fixed' else None)
                except Exception as error:r=dict(answer='',trace=[],fallback=1,eval_status='runtime_failure',error=repr(error),traceback=traceback.format_exc())
                if protocol=='fixed' and cache:
                    import torch
                    torch.save(cache,cachefile)
                calls=g.calls[first:];r.update(qid=row['qid'],case_id=row['case_id'],variant=row['variant'],protocol=protocol,method=name,seed=spec.get('seed'),identity_hash=h,checkpoint_hash=spec.get('sha256'),imported=False,seconds=time.time()-start,llm_calls=len(calls),input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls),actual_requests=sum(not x['cache_hit'] for x in calls),max_prompt_tokens=max([x['input_tokens'] for x in calls] or [0]))
                write(path,r);outputs.append(r);print(scope,ex.qid,protocol,name,r['eval_status'],round(r['seconds'],2),flush=True);del neural
        return outputs
    with ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS','48'))) as pool:results=[r for batch in pool.map(task,rows) for r in batch]
    status=dict(scope=scope,records=len(results),expected=len(rows)*(9 if smoke else 12),computed=sum(not r['imported'] for r in results),imported=sum(r['imported'] for r in results),failures=sum(r['eval_status']!='ok' for r in results),hash=h)
    write(folder/'summary.json',status);return status


def report(c,smoke=False):
    root=pathlib.Path(c['paths']['workdir']);scope='smoke' if smoke else 'test';folder=root/'runs'/('diagnostics_'+scope);lock=json.loads((root/'manifests'/('diagnostics_'+scope+'_lock.json')).read_text());records=[json.loads(p.read_text()) for p in folder.glob('T_*.json')];gold=json.loads((root/'data/private/T.json').read_text());check=official_function(root,'check_answer');split=json.loads((root/'data/adapt/split.json').read_text());expected_cases=len(split['split']['dev'][:4]) if smoke else len(split['split']['test']);variants=1 if smoke else 3
    groups={r['case_id']:r['group'] for r in json.loads((root/'data/adapt/samples.json').read_text())}
    table=[];scores={};question_scores={};fixed_audit=[];failures=[]
    for r in records:
        assert r['identity_hash']==lock['hash'];r['correct']=bool(check(True,gold[str(r['case_id'])],r['answer']))
        if r['eval_status']!='ok' or r.get('fallback'):failures.append({k:r.get(k) for k in ['qid','protocol','method','eval_status','fallback','error']})
    for protocol,method in sorted({(r['protocol'],r['method']) for r in records}):
        rr=[r for r in records if (r['protocol'],r['method'])==(protocol,method)];cases=collections.defaultdict(list)
        for r in rr:cases[r['case_id']].append(r)
        scores[(protocol,method)]={q:float(any(r['correct'] for r in rs)) for q,rs in cases.items() if len(rs)==variants}
        question_scores[(protocol,method)]={r['qid']:float(r['correct']) for r in rr}
        table.append(dict(protocol=protocol,method=method,questions=len(rr),complete_cases=len(scores[(protocol,method)]),expected_cases=expected_cases,case_accuracy=float(np.mean(list(scores[(protocol,method)].values()))) if scores[(protocol,method)] else None,question_accuracy=float(np.mean([r['correct'] for r in rr])),failures=sum(r['eval_status']!='ok' for r in rr),fallback_questions=sum(bool(r.get('fallback')) for r in rr),computed=sum(not r['imported'] for r in rr),input_tokens_total=sum(r.get('input_tokens',0) for r in rr if not r['imported']),max_prompt_tokens=max(r.get('max_prompt_tokens',0) for r in rr)))
    for qid in sorted({r['qid'] for r in records}):
        common=next((r for r in records if r['qid']==qid and r['method']=='common_reference'),None);rr={r['method']:r for r in records if r['qid']==qid and r['protocol']=='fixed'}
        if not common or len(rr)!=6:continue
        def raw(r):return digest([(t['query'],t['retrieved_documents'],t['visible_spans']) for t in r['trace']])
        raw_match=all(raw(r)==raw(common) for r in rr.values())
        candidate_match=len({digest([t.get('candidates') for t in r['trace']]) for m,r in rr.items() if m!='raw_reader'})==1
        base=common['final_prompt'];prompts=all(r.get('final_prompt','').split('\nUncertain current structure (not evidence): ')[0]==base for r in rr.values())
        fixed_audit.append(dict(qid=qid,same_raw=raw_match,same_candidates=candidate_match,same_reader_raw_prefix=prompts))
    pairs=[]
    comparisons=[('open','json_fewshot','open','json_zero'),('open','json_fewshot','open','common_reference'),('fixed','json_fewshot','fixed','json_zero'),('fixed','json_fewshot','fixed','raw_reader')]
    for protocol in ['open','fixed']:
        names=[(protocol,f'trained_rebind_s{s}') for s in [17,29,43]]
        if all(n in scores for n in names):
            shared=set.intersection(*(set(scores[n]) for n in names));scores[(protocol,'trained_mean3')]={q:float(np.mean([scores[n][q] for n in names])) for q in shared}
            qs=set.intersection(*(set(question_scores[n]) for n in names));question_scores[(protocol,'trained_mean3')]={q:float(np.mean([question_scores[n][q] for n in names])) for q in qs}
            comparisons.append((protocol,'json_fewshot',protocol,'trained_mean3'))
    comparisons += [('fixed','trained_mean3','fixed','raw_reader'),('fixed','trained_mean3','open','trained_mean3')]
    qgroup={r['qid']:groups[r['case_id']] for r in records}
    for pa,ma,pb,mb in comparisons:
        if (pa,ma) not in scores or (pb,mb) not in scores:continue
        for metric,source in [('case',scores),('question',question_scores)]:
            aa=source[(pa,ma)];bb=source[(pb,mb)];gg=collections.defaultdict(list)
            for q in sorted(aa.keys()&bb.keys()):gg[groups[q] if metric=='case' else qgroup[q]].append(aa[q]-bb[q])
            if not gg:continue
            sizes=np.array([len(v) for v in gg.values()]);sums=np.array([sum(v) for v in gg.values()]);draw=np.random.default_rng(612).integers(0,len(gg),(2000,len(gg)));boot=sums[draw].sum(1)/sizes[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);pairs.append(dict(treatment=pa+'/'+ma,control=pb+'/'+mb,metric=metric,n=int(sizes.sum()),groups=len(gg),difference=float(sums.sum()/sizes.sum()),ci_low=float(lo),ci_high=float(hi)))
    reports=root/'reports';prefix='diagnostics_'+scope
    for name,rs in [('results',table),('paired',pairs)]:
        with (reports/(prefix+'_'+name+'.csv')).open('w') as f:
            if rs:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    write(reports/(prefix+'_fixed_audit.json'),fixed_audit);write(reports/(prefix+'_failures.json'),failures)
    with (reports/(prefix+'_predictions.jsonl')).open('w') as f:
        for r in records:f.write(json.dumps({k:r.get(k) for k in ['qid','case_id','variant','protocol','method','seed','answer','correct','eval_status','fallback','imported','identity_hash']})+'\n')
    complete=len(records)==expected_cases*variants*(9 if smoke else 12);unchanged=all(digest(root/'src/rebind_mvp'/name)==sha for name,sha in lock['identity']['sources'].items()) and digest(inspect.getsource(run))==lock['identity']['runner'] and digest(inspect.getsource(FewShotGenerator))==lock['identity']['wrapper']
    unchanged=unchanged and digest(c)==lock['identity']['config_hash'] and digest(root/'manifests/demonstrations.json')==lock['identity']['demonstrations_hash']
    unchanged=unchanged and all(digest(root/'data'/p)==lock['identity'][key] for p,key in [('public/T.json','public_hash'),('adapt/split.json','split_hash'),('memory/T.json','memory_hash'),('memory/T_e5.npy','vectors_hash')])
    unchanged=unchanged and all(digest(pathlib.Path(p))==sha for p,sha in lock['identity']['references'].items())
    status=dict(status='complete' if complete and unchanged else 'in_progress',scope=scope,records=len(records),expected=expected_cases*variants*(9 if smoke else 12),inference_unchanged=unchanged,fixed_audited=len(fixed_audit),raw_mismatches=sum(not r['same_raw'] for r in fixed_audit),candidate_mismatches=sum(not r['same_candidates'] for r in fixed_audit),reader_prefix_mismatches=sum(not r['same_reader_raw_prefix'] for r in fixed_audit),failures=sum(r['eval_status']!='ok' for r in records),hash=lock['hash'])
    write(reports/(prefix+'_completion.json'),status)
    lines=['# Few-shot and same-evidence diagnostics','',json.dumps(status), '', 'Exploratory follow-up on a previously evaluated holdout. No new training. Three demonstrations from distinct training edit groups; no downstream answers in demonstrations. Open COMMON and trained ReBind outputs are imported unchanged references; zero/few-shot JSON are rerun. In fixed evidence, six methods share the COMMON raw trajectory and all structured methods share a cached candidate schedule. Raw reader also represents omitting the added structured state. This is SAME evidence, not oracle-gold evidence.','', '| Protocol | Method | Questions | Case accuracy | Question accuracy | Failures | Fallbacks |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['protocol']} | {r['method']} | {r['questions']} | {r['case_accuracy']} | {r['question_accuracy']:.4f} | {r['failures']} | {r['fallback_questions']} |")
    lines+=['','The final ReBind reader already receives raw documents. Fixed-evidence degradation can implicate added state or its interface; it does not uniquely prove neural information compression. Few-shot gains likewise do not establish that the neural operator is unnecessary. Inspect candidate/raw-prefix audits and confidence intervals, including all failures. Demonstrations add prompt tokens within the same context ceiling; compare token cost rather than calling the budgets equal in total tokens.','',f'[Paired intervals]({prefix}_paired.csv) | [Fixed audit]({prefix}_fixed_audit.json) | [Failures]({prefix}_failures.json) | [Demonstrations](../manifests/demonstrations.json)']
    (reports/(prefix.upper()+'.md')).write_text('\n'.join(lines)+'\n');return status


if __name__=='__main__':
    import yaml,sys
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','report']);p.add_argument('--smoke',action='store_true');a=p.parse_args();c=yaml.safe_load(pathlib.Path('configs/diagnostics.yaml').read_text());start=time.time();status='ok'
    try:print(json.dumps(prepare(c) if a.phase=='prepare' else run(c,a.smoke) if a.phase=='run' else report(c,a.smoke)),flush=True)
    except BaseException:status='failed';raise
    finally:append(pathlib.Path(c['paths']['workdir'])/'runs/diagnostics/commands.jsonl',dict(argv=sys.argv,start=start,seconds=time.time()-start,status=status,code_hash=digest(pathlib.Path(__file__))))
