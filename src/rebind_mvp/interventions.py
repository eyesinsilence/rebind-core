import json,pathlib
import torch
from .source_reader import ReBindModule
from .audit import digest,write,append,Blocked

def run_interventions(c):
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); records=[]
    checkpoint=data/'diagnostic_checkpoints_v4/rebind/17/best.pt'
    if not checkpoint.exists(): raise Blocked('No trained diagnostic checkpoint available')
    examples=torch.load(data/'transitions/controlled_micro.pt',weights_only=True); model=ReBindModule(d=c['rebind']['hidden_dim'],layers=c['rebind']['layers']).eval().to('cuda:2'); model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda:2')['model'])
    with torch.no_grad():
        for example in examples[12:]:
            before={k:v.to('cuda:2') for k,v in example['before'].items()}; after={k:v.to('cuda:2') for k,v in example['after'].items()}
            old=model(**before); normal=model(**after)
            oldmask=torch.tensor([True,False],device='cuda:2'); scores=normal['source'].clone(); values=normal['source_values'].clone(); scores[0]=old['source'][0]; values[0]=old['source_values'][0]
            frozen=model(**after,freeze=(oldmask,scores,values,torch.cat([old['source_special'],normal['source_special'][1:]],0)))
            row=dict(group_id=example['group_id'],scope='controlled_inference_lesion_only',checkpoint_hash=digest(checkpoint),label=example['target'],before_prediction=int(old['unary'][0].argmax()),normal_prediction=int(normal['unary'][0].argmax()),frozen_old_source_prediction=int(frozen['unary'][0].argmax()),old_source_score_delta=(normal['source'][0]-old['source'][0]).cpu().tolist(),open_queries_run=False)
            records.append(row);append(root/'reports/interventions.jsonl',row)
    write(root/'reports/interventions_summary.json',dict(records=records,open_retrieval_fork_status='not_run',interpretation='Inference lesion does not establish H2 or H3; no retrieval occurs in this diagnostic.'))
    return run_natural_forks(c,records)


def run_natural_forks(c,controlled):
    from .proposal import Generator
    from .retrieval import E5,Retriever
    from .transitions import NeuralState
    from .evaluate import run_loop
    from .stages import official_scores
    from .schema import InferenceExample
    from .data import read_rows
    import concurrent.futures,copy,time
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); checkpoint=data/'checkpoints/rebind/17/best.pt'
    if not checkpoint.exists(): return dict(status='controlled_only',controlled=controlled,natural_forks='checkpoint_missing')
    directory=root/'runs/intervention_forks'; directory.mkdir(exist_ok=True); generator=Generator(c); encoder=E5(c['models']['retriever_path'],device='cuda:3'); pairs=[]
    for ds in c['retrieval']['datasets']:
        retriever=Retriever(c,ds,encoder); rows=list(read_rows(data/'public'/ds/'eval.jsonl'))[:16]; private={r['qid']:r for r in read_rows(data/'private'/ds/'eval.jsonl')}
        def process(row):
            ex=InferenceExample(**row); branches={}
            for branch in ['rebind','frozen_old_read']:
                local=copy.copy(retriever); local.calls=[]; state=NeuralState(encoder,c,checkpoint,branch,split="eval")
                generator.context=dict(qid=ex.qid,split='eval',protocol='predeclared_first16_fork',source_hash=digest(root/'manifests/sources.lock.json'))
                try:
                    pred=run_loop(ex,local,generator,c,branch,state)
                except Exception as e:
                    import traceback
                    pred=dict(qid=ex.qid,method=branch,answer='',eval_status='runtime_failure',trace=[],error=str(e),traceback=traceback.format_exc())
                pred.update(dataset=ds,checkpoint_hash=digest(checkpoint),**official_scores(root,ds,pred['answer'],private[ex.qid]))
                write(directory/(ds+'_'+ex.qid+'_'+branch+'.json'),pred);branches[branch]=pred
            normal,frozen=[branches[k] for k in ['rebind','frozen_old_read']]
            prefix=lambda r:[(t['query'],t['retrieved_ids']) for t in r['trace'][:2]]
            shared=len(normal['trace'])>=2 and len(frozen['trace'])>=2 and prefix(normal)==prefix(frozen)
            support=set(private[ex.qid]['supporting_doc_ids'])
            def observed(r,start):
                return {d['parent_doc_id'] for t in r['trace'][start:] for d in t['retrieved_documents'] if d['doc_id'] in t['seen_ids']}
            initial=observed(dict(trace=normal['trace'][:1]),0)
            pair=dict(dataset=ds,qid=ex.qid,selection='first16_of_locked_eval_order_no_gold_selection',shared_prefix_and_new_evidence=shared,freeze_rule='all_first_exposure_old_source_scores_values_special_logits_candidate_ID_remapped',normal_em=normal['em'],frozen_em=frozen['em'],normal_f1=normal['f1'],frozen_f1=frozen['f1'],queries_diverged=[t['query'] for t in normal['trace'][2:]]!=[t['query'] for t in frozen['trace'][2:]],normal_new_gold_support_read=len((observed(normal,2)-initial)&support),frozen_new_gold_support_read=len((observed(frozen,2)-initial)&support),eligible_verified_revision=None,interpretation='inference lesion with distribution shift; no source-role correctness labels',normal_path=str(directory/(ds+'_'+ex.qid+'_rebind.json')),frozen_path=str(directory/(ds+'_'+ex.qid+'_frozen_old_read.json')))
            pairs.append(pair);print('fork',ds,ex.qid,shared,normal['em'],frozen['em'],flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(process,rows))
    result=dict(controlled=controlled,pairs=pairs,natural_forks='executed',verified_revision_count=None,limitations=['This is an inference lesion, not a retrained ablation.','Numeric old-source changes and query divergence do not verify correct role/scope revision.'])
    write(root/'reports/interventions_summary.json',result);return result
