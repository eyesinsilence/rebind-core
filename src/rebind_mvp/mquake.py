from .runtime import resolve_device,encoder_dimension,peak_memory
"""Frozen-checkpoint transfer to the official all-edited MQuAKE-Remastered-T task."""
import ast,collections,csv,json,os,pathlib,time,types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .audit import digest,write,append
from .data import read_rows
from .schema import InferenceExample


def official_function(root,name):
    path=root/'upstream/mquake_remastered'/('data_utils.py' if name=='check_answer' else 'eval/mquake_remastered/mquake_dataset.py')
    tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def prepare(c):
    import pyarrow.parquet as pq,subprocess
    root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root']);mask=official_function(root,'get_edits_without_contamination');audit={}
    for split in ['T','CF3k']:
        raw=data/'raw'/(split+'-00000-of-00001.parquet');rows=pq.read_table(raw).to_pylist()
        dataset=types.SimpleNamespace(dataset=rows,rand_list=[r['case_id'] for r in rows])
        texts=sorted({e['prompt'].format(e['subject'])+' '+e['target_new_str'] for r in rows for e in r['requested_rewrite']})
        ids={text:'update_'+digest(text)[:20] for text in texts}
        docs=[dict(doc_id=ids[text],parent_doc_id=ids[text],title='Supplied fact update',text=text,offsets=[0,len(text)]) for text in texts]
        write(data/'memory'/(split+'.json'),docs)
        selected=rows if split=='T' else rows[:c['mquake']['smoke_cases']]
        public=[];private={};excluded=[];own_missing=[]
        for r in selected:
            allowed,_,_,_=mask(dataset,r);allowed_ids=sorted({ids[text] for text in allowed})
            own={ids[e['prompt'].format(e['subject'])+' '+e['target_new_str']] for e in r['requested_rewrite']}
            missing=own-set(allowed_ids)
            if missing:own_missing.append(dict(case_id=r['case_id'],doc_ids=sorted(missing)))
            excluded.append(len(docs)-len(allowed_ids))
            private[str(r['case_id'])]={k:r[k] for k in ['case_id','answer','answer_alias','new_answer','new_answer_alias']}
            private[str(r['case_id'])].update(edit_group=digest(sorted(own)),own_edit_doc_ids=sorted(own),hops=len(r['new_triples']))
            for variant,question in enumerate(r['questions']):
                public.append(dict(qid=f"{split}_{r['case_id']:06d}_q{variant}",case_id=r['case_id'],variant=variant,question=question,allowed_doc_ids=allowed_ids))
        write(data/'public'/(split+'.json'),public);write(data/'private'/(split+'.json'),private)
        audit[split]=dict(raw_cases=len(rows),evaluated_cases=len(selected),questions=len(public),unique_edit_facts=len(docs),hop_counts=dict(collections.Counter(len(r['new_triples']) for r in selected)),nonempty_split_labels=sum(any(r['split'].values()) for r in rows),masked_edits_min=min(excluded),masked_edits_max=max(excluded),masked_edits_mean=float(np.mean(excluded)),own_edit_missing=own_missing,raw_sha256=digest(raw),public_sha256=digest(data/'public'/(split+'.json')),private_sha256=digest(data/'private'/(split+'.json')),memory_sha256=digest(data/'memory'/(split+'.json')))
    audit.update(dataset_revision=c['mquake']['dataset_revision'],official_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'upstream/mquake_remastered',text=True).strip(),edit_setting='All case IDs explicitly edited. Downloaded split columns are empty, so partial-edit assignments are unavailable.',masking='Exact official function: private gold paths are used ONLY to exclude contaminating updates during offline preprocessing. No gold path, unchanged-hop answer, or decomposition is exposed to inference.',deduplication='Identical supplied update strings are deduplicated before E5 retrieval; duplicate case membership is not exposed as evidence.',training='None on MQuAKE. Existing 2Wiki weak-supervision seed17 BP/ReBind checkpoints reused unchanged.',metric='Exact upstream case-insensitive answer/alias equality. Case success = any of three independently answered paraphrases; also report each-question accuracy. No answer-dependent early stopping.',snapshot='Current means the supplied benchmark edited world, not live current events.')
    write(root/'reports/mquake_data_audit.json',audit)
    return audit


class EditRetriever:
    def __init__(self,docs,vectors,allowed,encoder,k):
        allowed=set(allowed);keep=[i for i,d in enumerate(docs) if d['doc_id'] in allowed]
        self.docs=[docs[i] for i in keep];self.vectors=vectors[keep];self.encoder=encoder;self.k=k;self.calls=[]
    def search(self,question):
        vector=self.encoder.encode([question],'query')[0];scores=self.vectors@vector
        order=np.argsort(-scores,kind='stable')[:self.k];found=[dict(self.docs[i],score=float(scores[i])) for i in order]
        self.calls.append(dict(query=question,doc_ids=[d['doc_id'] for d in found]));return found


def protocol_lock(c):
    import inspect
    root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root'])
    excluded={'mquake.py','cli.py','stages.py','data.py','audit.py','train.py','diagnostics.py','interventions.py'}
    pipeline=dict(files={str(p.relative_to(root)):digest(p) for p in (root/'src').rglob('*.py') if p.name not in excluded},runner=inspect.getsource(evaluate),retriever=inspect.getsource(EditRetriever))
    return dict(config_hash=digest(c),pipeline_hash=digest(pipeline),data_audit_hash=digest(root/'reports/mquake_data_audit.json'),public_hash=digest(data/'public/T.json'),memory_hash=digest(data/'memory/T.json'),checkpoints={m:digest(pathlib.Path(c['mquake']['checkpoint_root'])/m/'17/best.pt') for m in ['bp_rebind','rebind']})


def evaluate(c,args):
    from .proposal import Generator
    from .retrieval import E5
    from .transitions import NeuralState
    from .evaluate import run_loop
    root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root']);split='CF3k' if args.phase=='smoke' else 'T';protocol=args.protocol
    methods=args.methods or c['mquake']['primary_methods' if protocol=='open' else 'fixed_methods']
    assert set(methods)<=set(c['mquake']['primary_methods']+c['mquake']['fixed_methods'])
    lock=protocol_lock(c);lockfile=root/'manifests/mquake_benchmark_lock.json'
    if args.phase=='evaluate':
        if lockfile.exists():assert json.loads(lockfile.read_text())['identity']==lock,'Locked MQuAKE inputs changed'
        else:
            assert (root/'runs/mquake_CF3k_open/summary.json').exists(),'Run separate CF smoke first'
            write(lockfile,dict(identity=lock,locked_at=time.time(),scope='Full T; frozen 2Wiki seed17 transfer; no T-driven selection'))
    rows=json.loads((data/'public'/(split+'.json')).read_text())
    if args.phase=='smoke' or protocol=='fixed':rows=[r for r in rows if r['variant']==c['mquake']['fixed_variant']]
    docs=json.loads((data/'memory'/(split+'.json')).read_text());encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever'))
    index=data/'memory'/(split+'_e5.npy');index_manifest=index.with_suffix('.json')
    fingerprint=digest(dict(memory=docs,encoder=c['models']['retriever_path'],format='title_newline_text_normalized_e5_v1'))
    if index.exists():assert json.loads(index_manifest.read_text())['input_hash']==fingerprint
    else:
        vectors=encoder.encode([d['title']+'\n'+d['text'] for d in docs]);np.save(index,vectors);write(index_manifest,dict(input_hash=fingerprint,vector_hash=digest(index),rows=len(docs)))
    vectors=np.load(index);assert np.max(abs(np.linalg.norm(vectors,axis=1)-1))<1e-5
    generator=Generator(c);folder=root/'runs'/f'mquake_{split}_{protocol}';folder.mkdir(parents=True,exist_ok=True)
    def question(row):
        ex=InferenceExample(qid=row['qid'],question=row['question']);results=[]
        for method in methods:
            file=folder/(ex.qid+'_'+method+'.json')
            if args.resume and file.exists():
                old=json.loads(file.read_text());assert old['identity']==lock;results.append(old);continue
            generator.context=dict(qid=ex.qid,split=split,protocol=f'mquake_{split}_{protocol}',source_hash=lock['data_audit_hash'])
            start=time.time();first=len(generator.calls);retriever=EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k'])
            checkpoint=pathlib.Path(c['mquake']['checkpoint_root'])/method/'17/best.pt';neural=None
            if method in ['bp_rebind','rebind']:neural=NeuralState(encoder,c,checkpoint,method,split=split)
            try:
                if method=='closed_book':
                    final=generator.json('Answer using your own knowledge without external facts. Return only JSON {"answer":"short answer","citations":[]}.\nQuestion: '+ex.question)
                    assert isinstance(final['answer'],str)
                    result=dict(answer=final['answer'],citations=[],trace=[],fallback=0,eval_status='ok')
                else:
                    fixed=None
                    if protocol=='fixed':
                        common=json.loads((root/'runs'/f'mquake_{split}_open'/(ex.qid+'_ircot_common.json')).read_text());assert common['identity']==lock
                        fixed=common['trace'];assert fixed,'COMMON produced no replayable evidence'
                    result=run_loop(ex,retriever,generator,c,method,neural,fixed)
            except Exception as error:
                import traceback
                result=dict(answer='',citations=[],trace=[],fallback=1,eval_status='runtime_failure',error=repr(error),traceback=traceback.format_exc())
            calls=generator.calls[first:]
            result.update(qid=ex.qid,case_id=row['case_id'],variant=row['variant'],method=method,dataset=split,protocol=protocol,identity=lock,checkpoint_hash=lock['checkpoints'].get(method),training_seed=17 if neural else None,seconds=time.time()-start,llm_calls=len(calls),input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls),cache_hits=sum(x['cache_hit'] for x in calls),actual_requests=sum(not x['cache_hit'] for x in calls),model_response_keys=[x['key'] for x in calls],retrieval_calls=retriever.calls)
            write(file,result);results.append(result);print(split,protocol,ex.qid,method,result['eval_status'],round(result['seconds'],2),flush=True)
        return results
    with ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS','32'))) as executor:
        results=[r for batch in executor.map(question,rows) for r in batch]
    summary=dict(split=split,protocol=protocol,questions=len(rows),methods=methods,predictions=len(results),failures=sum(r['eval_status']!='ok' for r in results),identity=lock)
    write(folder/'summary.json',summary);return summary


def report(c):
    root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root']);reports=root/'reports';check=official_function(root,'check_answer');records=[];tables=[];paired=[]
    label={s:json.loads((data/'private'/(s+'.json')).read_text()) for s in ['T','CF3k']}
    for folder in sorted((root/'runs').glob('mquake_*_*')):
        for path in sorted(folder.glob('*.json')):
            r=json.loads(path.read_text())
            if 'qid' not in r:continue
            gold=label[r['dataset']][str(r['case_id'])];seen={d for step in r['trace'] for d in step.get('seen_ids',[])}
            structural=[t for t in r['trace'] if 'candidates' in t]
            eligible=sum(len(t['candidates'])>=3 and any(len(v)>1 for v in t['candidates'].values()) for t in structural)
            r.update(correct=bool(check(True,gold,r['answer'])),old_correct=bool(check(False,gold,r['answer'])),own_update_read=bool(seen&set(gold['own_edit_doc_ids'])),edit_group=gold['edit_group'],hops=gold['hops'],nominal_triangle_fraction=eligible/len(structural) if structural else None,artifact=str(path.relative_to(root)));records.append(r)
    for split,protocol,method in sorted({(r['dataset'],r['protocol'],r['method']) for r in records}):
        rr=[r for r in records if (r['dataset'],r['protocol'],r['method'])==(split,protocol,method)];bycase=collections.defaultdict(list)
        for r in rr:bycase[r['case_id']].append(r)
        variants=3 if split=='T' and protocol=='open' else 1;complete=[r for r in bycase.values() if len(r)==variants];expected=len(label[split]);n=sum(len(r) for r in complete)
        tables.append(dict(dataset=split,protocol=protocol,method=method,expected_cases=expected,complete_cases=len(complete),missing_cases=expected-len(complete),questions=len(rr),question_accuracy=float(np.mean([r['correct'] for r in rr])),case_any_accuracy=float(np.mean([any(r['correct'] for r in group) for group in complete])) if complete else None,all_paraphrases_accuracy=float(np.mean([all(r['correct'] for r in group) for group in complete])) if complete else None,old_answer_persistence=float(np.mean([r['old_correct'] and not r['correct'] for r in rr])),own_update_read_rate=float(np.mean([r['own_update_read'] for r in rr])),failures=sum(r['eval_status']!='ok' for r in rr),fallback_questions=sum(bool(r.get('fallback')) for r in rr),llm_calls=float(np.mean([r['llm_calls'] for r in rr])),input_tokens=float(np.mean([r['input_tokens'] for r in rr])),actual_requests=sum(r['actual_requests'] for r in rr),seconds_mean=float(np.mean([r['seconds'] for r in rr])),scope='official any-of-3 case metric' if variants==3 else 'first-paraphrase diagnostic'))
    for protocol in ['open','fixed']:
        variants=3 if protocol=='open' else 1;by_method=collections.defaultdict(lambda:collections.defaultdict(list))
        for r in records:
            if r['dataset']=='T' and r['protocol']==protocol:by_method[r['method']][r['case_id']].append(r)
        treatment=by_method.get('rebind',{})
        for control,baseline in by_method.items():
            if control=='rebind':continue
            ids=sorted(q for q in treatment.keys()&baseline.keys() if len(treatment[q])==variants and len(baseline[q])==variants)
            if not ids:continue
            groups=collections.defaultdict(list)
            for q in ids:groups[treatment[q][0]['edit_group']].append(float(any(r['correct'] for r in treatment[q]))-float(any(r['correct'] for r in baseline[q])))
            sizes=np.array([len(v) for v in groups.values()]);sums=np.array([sum(v) for v in groups.values()]);rng=np.random.default_rng(612);draw=rng.integers(0,len(groups),(2000,len(groups)));boot=sums[draw].sum(1)/sizes[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
            paired.append(dict(protocol=protocol,control=control,n=len(ids),edit_groups=len(groups),difference=float(sums.sum()/sizes.sum()),ci_low=float(lo),ci_high=float(hi),resamples=2000))
    hops=[]
    for method in c['mquake']['primary_methods']:
        for hop in [2,3,4]:
            rr=[r for r in records if r['dataset']=='T' and r['protocol']=='open' and r['method']==method and r['hops']==hop];cases=collections.defaultdict(list)
            for r in rr:cases[r['case_id']].append(r)
            complete=[v for v in cases.values() if len(v)==3];nominal=[r['nominal_triangle_fraction'] for r in rr if r['nominal_triangle_fraction'] is not None]
            if rr:hops.append(dict(method=method,hops=hop,questions=len(rr),complete_cases=len(complete),question_accuracy=float(np.mean([r['correct'] for r in rr])),case_any_accuracy=float(np.mean([any(r['correct'] for r in v) for v in complete])) if complete else None,nominal_triangle_fraction=float(np.mean(nominal)) if nominal else None))
    for name,rows in [('mquake_results.csv',tables),('mquake_paired.csv',paired),('mquake_hops.csv',hops)]:
        with (reports/name).open('w') as f:
            if rows:
                out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    with (reports/'mquake_predictions.jsonl').open('w') as f:
        for r in records:f.write(json.dumps({k:r.get(k) for k in ['qid','case_id','variant','method','dataset','protocol','answer','correct','old_correct','own_update_read','eval_status','fallback','checkpoint_hash','training_seed','identity','llm_calls','actual_requests','artifact']},ensure_ascii=False)+'\n')
    write(reports/'mquake_failures.json',[{k:r.get(k) for k in ['qid','method','dataset','protocol','eval_status','fallback','error','artifact']} for r in records if r['eval_status']!='ok' or r.get('fallback')])
    checks=[]
    for case_id in label['T']:
        rr={r['method']:r for r in records if r['dataset']=='T' and r['protocol']=='fixed' and str(r['case_id'])==case_id}
        if not set(c['mquake']['fixed_methods'])<=rr.keys():continue
        raw={digest([(t['query'],t['retrieved_ids'],t['visible_spans']) for t in r['trace']]) for r in rr.values()}
        candidates={digest([t.get('candidates') for t in rr[m]['trace']]) for m in ['json_rebind','bp_rebind','rebind']}
        checks.append(dict(case_id=case_id,same_raw=len(raw)==1,same_candidates=len(candidates)==1))
    write(reports/'mquake_fixed_audit.json',dict(records=checks,n=len(checks),raw_matched=sum(r['same_raw'] for r in checks),candidates_matched=sum(r['same_candidates'] for r in checks)))
    lockpath=root/'manifests/mquake_benchmark_lock.json';unchanged=lockpath.exists() and json.loads(lockpath.read_text())['identity']==protocol_lock(c)
    expected=len(label['T'])*19;actual=sum(r['dataset']=='T' for r in records);complete=actual==expected and all(r['missing_cases']==0 for r in tables if r['dataset']=='T')
    status=dict(status='complete' if complete else 'in_progress',expected_predictions=expected,actual_predictions=actual,locked_inputs_unchanged=unchanged,training='frozen 2Wiki seed17 transfer; no MQuAKE training',tests='runs/initial_20260914/mquake_tests.log')
    write(reports/'mquake_completion.json',status)
    lines=['# MQuAKE-Remastered-T 迁移测试','',f"状态：{status['status']}；T 预测 {actual}/{expected}；封存输入未变：{unchanged}。",'', '使用此前 2Wiki 弱监督训练选定的 BP/REBIND seed17 checkpoint，Qwen2.5-7B 与 E5 冻结。没有在 T 上训练、选 checkpoint 或调参。16 个 CF3k 案例仅用于独立工程 smoke。','', '## 数据与口径','', '[数据审计](./mquake_data_audit.json) · [官方仓库](https://github.com/henryzhongsc/MQuAKE-Remastered) · [官方数据](https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered)','', 'T 共 1,864 个案例，每例三个问法，全部编辑设定。编辑库为 96 个去重后的更新事实，未加入 gold 多跳链或未编辑单跳答案。官方排除污染编辑的函数使用私有正确路径预处理候选编辑库：这是 benchmark 的标签感知过滤，不能声称完全 label-blind 的开放检索。当前下载版本 split 栏为空，因此未冒称复现 100/500-edit 子设置。','', '开放主表的 case accuracy 为三个独立问法至少一个正确，采用官方大小写不敏感的字符串/别名精确匹配；所有问法都执行，不根据答案提前结束。另报逐问准确率和三个问法全对率。同文诊断固定取每例第一个问法，重放 COMMON 的原文日程，不与 any-of-3 分数直接比较。','', '## 结果','', '| 数据 | 协议 | 方法 | 完整案例 | 逐问准确率 | 案例准确率 | 回退题数 |','|---|---|---|---:|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['dataset']} | {r['protocol']} | {r['method']} | {r['complete_cases']}/{r['expected_cases']} | {r['question_accuracy']:.4f} | {r['case_any_accuracy'] if r['case_any_accuracy'] is not None else 'N/A'} | {r['fallback_questions']} |")
    lines+=['','[完整结果表](./mquake_results.csv) · [逐题索引](./mquake_predictions.jsonl) · [配对区间](./mquake_paired.csv) · [同文核对](./mquake_fixed_audit.json)','', '## 判断边界','', '该任务允许冻结模型补全未编辑知识，并要求显式更新覆盖旧知识；因此相对百科检索实验，任务指令也必须适配。候选绑定仍只能取自实际检索的原文。JSON 状态新增候选字符串枚举约束，修复此前嵌套值接口问题；它不是完全不改接口的单因素数据集消融。闭卷基线检验 Qwen 是否原本已知新答案；较高分数不能自动证明发生了知识修订。统计以共享编辑事实为组 bootstrap 2,000 次，避免把同一更新衍生的大量问题误当独立样本。','']
    for r in paired:lines.append(f"- {r['protocol']}，REBIND 对 {r['control']}：差 {r['difference']:+.4f}，95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]；{r['n']} 案例、{r['edit_groups']} 个编辑组。")
    lines+=['','这轮首先检验已有实现的跨任务适用性；无论结果好坏，都不能单凭更换数据集将此前退化归因于数据集。来源作用域修订、候选覆盖与训练监督不足仍需分别验证。','', '[真实命令账本](../runs/initial_20260914/commands.jsonl) · [失败与回退](./mquake_failures.json) · [完成状态](./mquake_completion.json)','', '复现入口：`bash scripts/run_all.sh --resume`。所有模型与 checkpoint 从原实验路径复用，原报告和原封存结果保持独立。']
    lines+=['','[按跳数分层](./mquake_hops.csv)：T 中 1,421 例为两跳、441 例为三跳、四跳仅 2 例；四跳不能支持稳定总体判断。nominal_triangle_fraction 只表示至少三个变量且存在已知候选，不证明正确三角推理。T 提供显式事实更新，未直接提供旧来源作用域改判标签，最终答案改变不能单独证明来源重解释。']
    (reports/'MQUAKE_REPORT.md').write_text('\n'.join(lines)+'\n');return status


def run(c,args):
    if args.phase=='prepare':return prepare(c)
    if args.phase=='report':return report(c)
    return evaluate(c,args)


def adapt_evaluate(c):
    """Open QA on the train-disjoint holdout, after all checkpoint selections finish."""
    import inspect,torch,traceback
    from .proposal import Generator
    from .retrieval import E5
    from .transitions import NeuralState
    from .evaluate import run_loop
    root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root']);folder=root/'runs/adapt_qa';folder.mkdir(exist_ok=True)
    split=json.loads((data/'adapt/split.json').read_text());test=set(split['split']['test'])
    assert test.isdisjoint(split['split']['train']) and test.isdisjoint(split['split']['dev'])
    import yaml
    continuation=yaml.safe_load((root/'configs/adapt_continue.yaml').read_text())['adapt']
    selections={}
    for method in ['rebind','bp_rebind','independent_binding','no_revision_loss']:
        for seed in [17,29,43]:
            phase='checkpoints_continue' if [method,seed] in continuation['continue_pairs'] else 'checkpoints'
            directory=data/'adapt'/phase/method/str(seed)
            done=json.loads((directory/'complete.json').read_text());checkpoint=directory/'best.pt'
            selections[f'{method}_s{seed}']=dict(path=str(checkpoint),sha256=digest(checkpoint),best_epoch=done['best_epoch'],training_identity=done['identity'],selection='train-side dev loss only')
    methods=[dict(name=m,mode=m) for m in ['closed_book','ircot_common','json_rebind']]
    for method in ['bp_rebind','rebind']:
        for seed in [17,29,43]:
            frozen=pathlib.Path(c['mquake']['checkpoint_root'])/method/str(seed)/'best.pt'
            methods.extend([dict(name=f'frozen_{method}_s{seed}',mode=method,path=str(frozen),sha256=digest(frozen),seed=seed),dict(name=f'adapted_{method}_s{seed}',mode=method,seed=seed,**selections[f'{method}_s{seed}'])])
    identity=dict(config_hash=digest(c),split_hash=digest(data/'adapt/split.json'),public_hash=digest(data/'public/T.json'),memory_hash=digest(data/'memory/T.json'),vectors_hash=digest(data/'memory/T_e5.npy'),methods=methods,all_trained_selections=selections,inference_files={name:digest(root/'src/rebind_mvp'/name) for name in ['evaluate.py','proposal.py','transitions.py','source_reader.py','pair_update.py','bp.py','frontier.py','schema.py','retrieval.py']},runner=digest(inspect.getsource(adapt_evaluate)),retriever=digest(inspect.getsource(EditRetriever)))
    lockfile=root/'manifests/adapt_qa_lock.json'
    if lockfile.exists():assert json.loads(lockfile.read_text())['identity']==identity,'Adapted QA identity changed'
    else:write(lockfile,dict(identity=identity,time=time.time(),scope='292 train-disjoint holdout cases, three variants; no gold path in QA retrieval; 15 methods = 13,140 predictions'))
    rows=[r for r in json.loads((data/'public/T.json').read_text()) if r['case_id'] in test];assert len(rows)==len(test)*3
    docs=json.loads((data/'memory/T.json').read_text());vectors=np.load(data/'memory/T_e5.npy');encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever'));generator=Generator(c)
    identity_hash=digest(identity)
    def question(row):
        ex=InferenceExample(qid=row['qid'],question=row['question']);results=[]
        for spec in methods:
            path=folder/(ex.qid+'_'+spec['name']+'.json')
            if path.exists():
                old=json.loads(path.read_text());assert old['identity_hash']==identity_hash;results.append(old);continue
            generator.context=dict(qid=ex.qid,split='MQuAKE_T_adapt_holdout',protocol='open',source_hash=identity['memory_hash'])
            start=time.time();first=len(generator.calls);neural=None
            try:
                retriever=EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k'])
                if 'path' in spec:neural=NeuralState(encoder,c,pathlib.Path(spec['path']),spec['mode'],split='MQuAKE_T_adapt_holdout')
                if spec['mode']=='closed_book':
                    final=generator.json('Answer using your own knowledge without external facts. Return only JSON {"answer":"short answer","citations":[]}.\nQuestion: '+ex.question)
                    assert isinstance(final['answer'],str)
                    result=dict(answer=final['answer'],citations=[],trace=[],fallback=0,eval_status='ok')
                else:result=run_loop(ex,retriever,generator,c,spec['mode'],neural)
            except Exception as error:result=dict(answer='',trace=[],fallback=1,eval_status='runtime_failure',error=repr(error),traceback=traceback.format_exc())
            calls=generator.calls[first:]
            result.update(qid=row['qid'],case_id=row['case_id'],variant=row['variant'],method=spec['name'],seed=spec.get('seed'),identity_hash=identity_hash,checkpoint_hash=spec.get('sha256'),seconds=time.time()-start,llm_calls=len(calls),actual_requests=sum(not x['cache_hit'] for x in calls),input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls))
            write(path,result);results.append(result);print(row['qid'],spec['name'],result['eval_status'],round(result['seconds'],2),flush=True)
            del neural
        return results
    with ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS','48'))) as pool:results=[r for batch in pool.map(question,rows) for r in batch]
    summary=dict(expected=len(rows)*len(methods),actual=len(results),failures=sum(r['eval_status']!='ok' for r in results),identity_hash=identity_hash)
    write(folder/'summary.json',summary);return summary


def adapt_report(c):
    root=pathlib.Path(c['paths']['workdir']);data=pathlib.Path(c['paths']['data_root']);reports=root/'reports'
    split=json.loads((data/'adapt/split.json').read_text());test=set(split['split']['test']);gold=json.loads((data/'private/T.json').read_text());check=official_function(root,'check_answer')
    lock=json.loads((root/'manifests/adapt_qa_lock.json').read_text());identity_hash=digest(lock['identity']);methods=[s['name'] for s in lock['identity']['methods']];records=[];group_of={}
    for s in json.loads((data/'adapt/samples.json').read_text()):group_of[s['case_id']]=s['group']
    for path in sorted((root/'runs/adapt_qa').glob('*.json')):
        if path.name=='summary.json':continue
        r=json.loads(path.read_text());assert r['case_id'] in test and r['identity_hash']==identity_hash
        r.update(correct=bool(check(True,gold[str(r['case_id'])],r['answer'])),old_correct=bool(check(False,gold[str(r['case_id'])],r['answer'])),group=group_of[r['case_id']],artifact=str(path.relative_to(root)));records.append(r)
    tables=[];case_scores={};by_method=collections.defaultdict(list)
    for r in records:by_method[r['method']].append(r)
    for method in methods:
        rr=by_method[method];by_case=collections.defaultdict(list)
        for r in rr:by_case[r['case_id']].append(r)
        case_scores[method]={q:float(any(r['correct'] for r in by_case[q])) for q in test if len(by_case[q])==3}
        tables.append(dict(method=method,questions=len(rr),expected_questions=len(test)*3,complete_cases=len(case_scores[method]),question_accuracy=float(np.mean([r['correct'] for r in rr])) if rr else None,case_accuracy=float(np.mean(list(case_scores[method].values()))) if case_scores[method] else None,failures=sum(r['eval_status']!='ok' for r in rr),fallback_questions=sum(bool(r.get('fallback')) for r in rr)))
    pairs=[]
    comparisons=[]
    for method in ['bp_rebind','rebind']:
        for seed in [17,29,43]:comparisons.append((f'adapted_{method}_s{seed}',f'frozen_{method}_s{seed}'))
        for phase in ['frozen','adapted']:
            names=[f'{phase}_{method}_s{seed}' for seed in [17,29,43]]
            case_scores[f'{phase}_{method}_mean3']={q:float(np.mean([case_scores[n][q] for n in names])) for q in test if all(q in case_scores[n] for n in names)}
        comparisons.append((f'adapted_{method}_mean3',f'frozen_{method}_mean3'))
    comparisons += [('adapted_rebind_mean3',control) for control in ['closed_book','ircot_common','json_rebind','adapted_bp_rebind_mean3']]
    for treatment,control in comparisons:
        ids=sorted(case_scores[treatment].keys()&case_scores[control].keys());groups=collections.defaultdict(list)
        for q in ids:groups[group_of[q]].append(case_scores[treatment][q]-case_scores[control][q])
        if not groups:continue
        sizes=np.array([len(v) for v in groups.values()]);sums=np.array([sum(v) for v in groups.values()]);draw=np.random.default_rng(612).integers(0,len(groups),(2000,len(groups)));boot=sums[draw].sum(1)/sizes[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
        pairs.append(dict(treatment=treatment,control=control,cases=len(ids),groups=len(groups),difference=float(sums.sum()/sizes.sum()),ci_low=float(lo),ci_high=float(hi)))
    seed_summary=[]
    for label in ['closed_book','ircot_common','json_rebind','frozen_bp_rebind','adapted_bp_rebind','frozen_rebind','adapted_rebind']:
        chosen=[r for r in tables if r['method']==label or r['method'].startswith(label+'_s')]
        if not chosen or any(r['case_accuracy'] is None for r in chosen):continue
        seed_summary.append(dict(method=label,seeds=len(chosen),case_accuracy_mean=float(np.mean([r['case_accuracy'] for r in chosen])),case_accuracy_std=float(np.std([r['case_accuracy'] for r in chosen],ddof=1)) if len(chosen)>1 else None,question_accuracy_mean=float(np.mean([r['question_accuracy'] for r in chosen])),question_accuracy_std=float(np.std([r['question_accuracy'] for r in chosen],ddof=1)) if len(chosen)>1 else None))
    for name,rows in [('adapt_qa_results.csv',tables),('adapt_qa_paired.csv',pairs),('adapt_qa_seed_summary.csv',seed_summary)]:
        with (reports/name).open('w') as f:
            if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (reports/'adapt_qa_predictions.jsonl').open('w') as f:
        for r in records:f.write(json.dumps({k:r.get(k) for k in ['qid','case_id','variant','method','seed','answer','correct','old_correct','eval_status','fallback','checkpoint_hash','identity_hash','artifact']})+'\n')
    failures=[{k:r.get(k) for k in ['qid','method','eval_status','fallback','error','artifact']} for r in records if r['eval_status']!='ok' or r.get('fallback')];write(reports/'adapt_qa_failures.json',failures)
    unchanged=all(digest(pathlib.Path(s['path']))==s['sha256'] for s in lock['identity']['methods'] if 'path' in s)
    unchanged=unchanged and all(digest(root/'src/rebind_mvp'/name)==sha for name,sha in lock['identity']['inference_files'].items())
    import inspect
    unchanged=unchanged and digest(c)==lock['identity']['config_hash'] and digest(inspect.getsource(adapt_evaluate))==lock['identity']['runner'] and digest(inspect.getsource(EditRetriever))==lock['identity']['retriever']
    unchanged=unchanged and all(digest(data/path)==lock['identity'][key] for path,key in [('adapt/split.json','split_hash'),('public/T.json','public_hash'),('memory/T.json','memory_hash'),('memory/T_e5.npy','vectors_hash')])
    complete=all(r['questions']==r['expected_questions'] and r['complete_cases']==len(test) for r in tables)
    status=dict(status='complete' if complete and unchanged else 'in_progress',predictions=len(records),expected=len(test)*3*len(methods),checkpoint_and_core_unchanged=unchanged,train_test_disjoint=test.isdisjoint(split['split']['train']) and test.isdisjoint(split['split']['dev']),identity_hash=identity_hash)
    write(reports/'adapt_qa_completion.json',status)
    lines=['# MQuAKE 训练迁移后的开放问答','',f"状态：{status['status']}；预测 {len(records)}/{status['expected']}；检查点和核心推理代码未变：{unchanged}。",'', '292 个按编辑事实连通组隔离的训练保留案例、每例三问。候选库仅包含官方更新事实，没有把受控训练的完整事实链带入QA。主分数为三个问法任一答对的案例准确率，官方大小写不敏感精确答案/别名匹配。', '', '| 方法 | 预测数 | 完整案例 | 案例准确率 | 失败 | 回退 |','|---|---:|---:|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['method']} | {r['questions']} | {r['complete_cases']} | {r['case_accuracy']} | {r['failures']} | {r['fallback_questions']} |")
    lines+=['','## 配对变化','']
    for r in pairs:lines.append(f"- {r['treatment']} 对 {r['control']}：{r['difference']:+.4f}，编辑组 bootstrap 95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]，{r['cases']}案例/{r['groups']}组。")
    lines+=['','## 判断边界','', '修订训练的受控开发高分不能单独证明开放QA改善；结论以本表适配前后同方法同种子的配对结果为依据。COMMON/JSON 为不训练对照；闭卷用于检查模型原本已知答案的混杂。mean3 是每例三个训练种子成功率的均值，不是从三个种子中择优。保留集只有14个事实组，区间需谨慎解释。', '', '这292案例是在旧冻结评测部分运行后以固定划分规则重新指定的训练保留集，不冒称从未触碰的外部标准测试；划分与训练选点未使用旧QA正确性。训练组/开发组均未计入主分数。', '', '[训练报告](MQUAKE_TRAINING.md) · [完整结果](adapt_qa_results.csv) · [配对区间](adapt_qa_paired.csv) · [逐题索引](adapt_qa_predictions.jsonl) · [失败记录](adapt_qa_failures.json) · [封存身份](../manifests/adapt_qa_lock.json) · [真实命令](../runs/adapt_qa/commands.jsonl)']
    if complete:
        means={r['method']:r for r in seed_summary};old=means['frozen_rebind'];new=means['adapted_rebind']
        comparison=next(r for r in pairs if r['treatment']=='adapted_rebind_mean3' and r['control']=='frozen_rebind_mean3')
        conclusion=['## 实际结论','',f"补训后 ReBind 的三种子平均案例准确率由 {old['case_accuracy_mean']:.2%} 变为 {new['case_accuracy_mean']:.2%}（{100*comparison['difference']:+.2f} 个百分点）；编辑组配对 95% CI 为 [{100*comparison['ci_low']:+.2f}, {100*comparison['ci_high']:+.2f}] 个百分点。逐问准确率则由 {old['question_accuracy_mean']:.2%} 变为 {new['question_accuracy_mean']:.2%}。",'',f"COMMON 案例准确率 {means['ircot_common']['case_accuracy_mean']:.2%}，JSON {means['json_rebind']['case_accuracy_mean']:.2%}，闭卷 {means['closed_book']['case_accuracy_mean']:.2%}。本次补训出现小幅案例指标改善，但区间包含零、逐问指标下降，也未超过强对照；现有证据不能将原先退化单独归因于训练不足。",'', '受控修订接近满分而开放问答没有稳定优势，表明受控监督的改善尚未可靠迁移到实际检索轨迹；该现象与候选或角色分布差异等解释相容，但本评测没有因果隔离这些原因。','', '| 方法 | 种子数 | 案例准确率均值 | 逐问准确率均值 |','|---|---:|---:|---:|']
        for r in seed_summary:conclusion.append(f"| {r['method']} | {r['seeds']} | {r['case_accuracy_mean']:.2%} | {r['question_accuracy_mean']:.2%} |")
        conclusion += ['', f"完整保留 {len(failures)} 个发生回退或最终失败的方法-问题记录，其中最终运行/解析失败 {sum(r['eval_status']!='ok' for r in records)} 个；未删掉失败分母。种子标准差见 [种子汇总](adapt_qa_seed_summary.csv)。",'']
        lines[4:4]=conclusion
    (reports/'MQUAKE_ADAPTED_QA.md').write_text('\n'.join(lines)+'\n');return status


if __name__=='__main__':
    import argparse,yaml,sys
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['adapt-evaluate','adapt-report']);parser.add_argument('--config',default='configs/resolved.yaml');args=parser.parse_args();c=yaml.safe_load(pathlib.Path(args.config).read_text());start=time.time();status='ok'
    try:print(json.dumps(adapt_evaluate(c) if args.phase=='adapt-evaluate' else adapt_report(c)),flush=True)
    except BaseException:status='failed';raise
    finally:append(pathlib.Path(c['paths']['workdir'])/'runs/adapt_qa/commands.jsonl',dict(argv=sys.argv,start=start,seconds=time.time()-start,status=status,code_hash=digest(pathlib.Path(__file__))))
