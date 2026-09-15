"""Frozen-checkpoint development rerun after code fixes; no training or checkpoint selection."""
import argparse
import collections
import copy
import csv
import inspect
import json
import os
import pathlib
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import yaml

from rebind_mvp.audit import digest,write
from rebind_mvp.evaluate import run_loop
from rebind_mvp.final_frontend import run_repaired
from rebind_mvp.mquake import EditRetriever,official_function
from rebind_mvp.proposal import Generator
from rebind_mvp.retrieval import E5
from rebind_mvp.runtime import resolve_device
from rebind_mvp.schema import InferenceExample
from rebind_mvp.transitions import NeuralState
from audit_frontend_execution import audit as audit_execution


def run(config,partition,stage,seeds):
    c=copy.deepcopy(config);root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root'])
    manifest=json.loads((root/'manifests/split_manifest.json').read_text())
    selected=json.loads((root/'manifests/frozen_checkpoints.json').read_text())['selected_checkpoints']
    groups={case:g['group'] for g in manifest['groups'] for case in g['cases']}
    cases=manifest['dev_'+partition]
    if stage=='smoke':cases=[min(i for i in cases if groups[i]==g) for g in sorted({groups[i] for i in cases})]
    rows=[r for r in json.loads((data/'public/T.json').read_text()) if r['case_id'] in cases]
    specs=[dict(name='COMMON',mode='ircot_common'),dict(name='JSON',mode='json_fix')]
    for seed in seeds:
        for family,mode in [('BP','bp_rebind'),('ReBind','rebind')]:
            checkpoint=selected[f'{mode}_s{seed}']
            assert digest(pathlib.Path(checkpoint['path']))==checkpoint['sha256'],'Checkpoint mismatch'
            specs.append(dict(name=f'{family}_s{seed}',family=family,mode=mode,seed=seed,checkpoint=checkpoint))
    tag=f'{stage}_{partition}'
    folder=root/'runs/debug'/tag;folder.mkdir(parents=True,exist_ok=True)
    code={str(p.relative_to(root)):digest(p) for p in (root/'src').rglob('*.py')}
    c['component_versions']=dict(debug_code=digest(code),relation_policy=c['final_plan']['relation_policy'])
    identity=dict(config=c,rows=digest(rows),specs=specs,sources=code,runner=digest(pathlib.Path(__file__)),
                  memory=digest(data/'memory/T.json'),vectors=digest(data/'memory/T_e5.npy'),
                  split=digest(root/'manifests/split_manifest.json'),stage=tag,expected=len(rows)*len(specs))
    identity_hash=digest(identity);lock=folder/'identity.json'
    if lock.exists():assert json.loads(lock.read_text())['hash']==identity_hash,'Use a fresh run directory for changed code'
    else:write(lock,dict(hash=identity_hash,identity=identity))
    generator=Generator(c);encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever'))
    docs=json.loads((data/'memory/T.json').read_text());vectors=np.load(data/'memory/T_e5.npy')
    assert vectors.shape==(len(docs),encoder.embedding_dim),'Rebuild index for the chosen encoder'
    started=time.time()

    def question(row):
        results=[]
        for spec in specs:
            path=folder/(row['qid']+'_'+spec['name']+'.json')
            if path.exists():
                result=json.loads(path.read_text());assert result['identity_hash']==identity_hash
                results.append(result);continue
            generator.context=dict(qid=row['qid'],split=tag,method=spec['name'],components=c['component_versions'])
            start=len(generator.calls);retriever=EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k'])
            example=InferenceExample(row['qid'],row['question']);neural=None
            try:
                if spec['mode']=='ircot_common':result=run_loop(example,retriever,generator,c,'ircot_common')
                else:
                    if 'checkpoint' in spec:
                        neural=NeuralState(encoder,c,spec['checkpoint']['path'],spec['mode'],split=tag)
                        neural.checkpoint_hash=spec['checkpoint']['sha256']
                    result=run_repaired(example,retriever,generator,c,spec['mode'],neural,knowledge='hybrid')
            except Exception as error:
                result=dict(answer='',eval_status='runtime_failure',trace=[],error=repr(error),traceback=traceback.format_exc())
            calls=generator.calls[start:]
            result.update(qid=row['qid'],case_id=row['case_id'],variant=row['variant'],method=spec['name'],seed=spec.get('seed'),
                          identity_hash=identity_hash,checkpoint_hash=spec.get('checkpoint',{}).get('sha256'),
                          actual_retriever_calls=retriever.calls,actual_requests=sum(not x['cache_hit'] for x in calls),
                          input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls))
            write(path,result);results.append(result)
            print(tag,row['qid'],spec['name'],result['eval_status'],flush=True)
            del neural
        return results

    with ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS','16'))) as pool:
        records=[r for batch in pool.map(question,rows) for r in batch]
    assert len(records)==identity['expected']
    assert len({(r['qid'],r['method']) for r in records})==len(records)
    # All gold access occurs after inference. No outcome selects questions or checkpoints.
    gold=json.loads((data/'private/T.json').read_text());check=official_function(root,'check_answer')
    public={r['qid']:r for r in rows};predictions=[];stats=[]
    for record in records:
        row=public[record['qid']]
        assert [t['query'] for t in record['trace']]==[r['query'] for r in record['actual_retriever_calls']] or record['eval_status']=='runtime_failure'
        assert all(set(call['doc_ids'])<=set(row['allowed_doc_ids']) for call in record['actual_retriever_calls'])
        predictions.append({k:record.get(k) for k in ['qid','case_id','variant','method','seed','answer','eval_status','identity_hash','checkpoint_hash']} |
                           dict(correct=bool(check(True,gold[str(row['case_id'])],record['answer'])),group=groups[row['case_id']]))
    for spec in specs:
        p=[r for r in predictions if r['method']==spec['name']];rr=[r for r in records if r['method']==spec['name']]
        case_success=collections.defaultdict(list)
        for row in p:case_success[row['case_id']].append(row['correct'])
        stats.append(dict(method=spec['name'],questions=len(p),accuracy=float(np.mean([r['correct'] for r in p])),
                          case_accuracy=float(np.mean([any(v) for v in case_success.values()])),
                          runtime_failures=sum(r['eval_status']=='runtime_failure' for r in rr),
                          non_ok=sum(r['eval_status']!='ok' for r in rr),
                          graph_fallbacks=sum(bool(r.get('frontend_failure')) for r in rr),
                          requests=sum(r['actual_requests'] for r in rr),input_tokens=sum(r['input_tokens'] for r in rr),
                          output_tokens=sum(r['output_tokens'] for r in rr)))
    audit=audit_execution(r for r in records if r['method']!='COMMON')
    assert not audit['violations'],audit
    assert all(digest(root/p)==sha for p,sha in code.items()),'Code changed during inference'
    out=root/'reports/debug'/tag;out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.jsonl').open('w') as f:
        for row in predictions:f.write(json.dumps(row)+'\n')
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    summary=dict(stage=tag,records=len(records),expected=identity['expected'],results=stats,execution_audit=audit,
                 seconds=time.time()-started,identity_hash=identity_hash,training_updates=0,
                 status='complete' if not any(s['non_ok'] for s in stats) else 'completed_with_failures')
    write(out/'summary.json',summary);write(folder/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=pathlib.Path,required=True)
    parser.add_argument('--partition',choices=['build','check'],default='build')
    parser.add_argument('--stage',choices=['smoke','dev'],default='smoke')
    parser.add_argument('--seeds',nargs='+',type=int,default=[17])
    args=parser.parse_args();torch.set_num_threads(2)
    result=run(yaml.safe_load(args.config.read_text()),args.partition,args.stage,args.seeds)
    raise SystemExit(0 if result['status']=='complete' else 1)
