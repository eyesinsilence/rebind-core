"""Execution entry point; offline labels are read only inside audit/report/train phases."""
import argparse, collections, copy, datetime, inspect, json, pathlib, re, subprocess, sys, time
import yaml
from rebind_mvp.audit import digest, write, append


def audit(c):
    import pyarrow.parquet as pq
    import torch
    from rebind_mvp.source_reader import ReBindModule
    from rebind_mvp.joint_decoder import decode
    root=pathlib.Path(c['paths']['workdir']);parent=pathlib.Path(c['diagnostics']['parent']);baseline=pathlib.Path(c['final_plan']['baseline']);out=root/'reports'
    raw=pq.read_table(root/'data/raw/T-00000-of-00001.parquet').to_pylist();rows={r['case_id']:r for r in raw};split=json.loads((root/'data/adapt/split.json').read_text());samples=json.loads((root/'data/adapt/samples.json').read_text());groups={r['case_id']:r['group'] for r in samples};public=json.loads((root/'data/public/T.json').read_text());byqid={r['qid']:r for r in public}
    histories=collections.defaultdict(set)
    for project in [parent,baseline,pathlib.Path(c['frontier_diagnostics']['prior'])]:
        for folder in (project/'runs').iterdir():
            if not folder.is_dir():continue
            for p in folder.glob('T_*.json'):
                match=re.match(r'T_(\d+)_q\d',p.name)
                if match:histories[groups[int(match[1])]].add(str(folder))
    devgroups=sorted(split['groups']['dev'],key=lambda g:digest([20260915,g]));chosen=[];group_inventory=[]
    for name,gs in split['groups'].items():
        for g in gs:
            ids=[i for i in split['split'][name] if groups[i]==g]
            if name=='dev':
                strata=collections.defaultdict(list)
                for i in ids:strata[tuple(t[1] for t in rows[i]['new_triples'])].append(i)
                for v in strata.values():v.sort(key=lambda i:digest([20260915,i]))
                selected=[]
                while any(strata.values()) and len(selected)<c['final_plan']['per_dev_group']:
                    for k,v in sorted(strata.items()):
                        if v and len(selected)<c['final_plan']['per_dev_group']:selected.append(v.pop(0))
                chosen+=selected
            group_inventory.append(dict(group=g,original_split=name,cases=ids,checkpoint_training=name=='train',checkpoint_selection=name=='dev',controlled_evaluation=name=='test',prior_qa_directories=sorted(histories[g]),new_role=('dev_build' if devgroups.index(g)%2==0 else 'dev_check') if name=='dev' else 'D_seen' if name=='test' else 'train'))
    manifest=dict(seed=20260915,groups=group_inventory,dev_build=sorted(i for i in chosen if devgroups.index(groups[i])%2==0),dev_check=sorted(i for i in chosen if devgroups.index(groups[i])%2==1),D_seen=split['split']['test'],D_fresh=[],D_fresh_reason='All 93 edit groups participated in checkpoint training, checkpoint selection or controlled/QA evaluation. No untouched edit group remains.',dev_check_scope='Disjoint groups for new frontend decisions; already used for historical checkpoint selection, not an untouched test set.',all_variants_together=True,selection_uses_correctness=False,source_split_hash=digest(root/'data/adapt/split.json'))
    write(root/'manifests/split_manifest.json',manifest)
    memory=json.loads((root/'data/memory/T.json').read_text());normalize=lambda s:' '.join(s.casefold().strip().rstrip('.').split());memory_by_text={normalize(d['text']):d['doc_id'] for d in memory};triples={};mapped=set()
    for r in raw:
        for request,trip in zip(r['requested_rewrite'],r['edit_triples']):
            text=request['prompt'].format(request['subject'])+' '+request['target_new_str'];did=memory_by_text.get(normalize(text))
            if did:triples[tuple(trip)]=did;mapped.add(did)
    bridge=[]
    for i in chosen:
        r=rows[i];allowed=set(byqid[f'T_{i:06d}_q0']['allowed_doc_ids'])
        for j,(old,new) in enumerate(zip(r['orig_triples'],r['new_triples'])):
            did=triples.get(tuple(new));present=did in allowed
            category=('edited_memory_present' if old!=new else 'unedited_memory_present') if present else 'not_in_memory' if len(mapped)==len(memory) else 'unknown'
            bridge.append(dict(case_id=i,group=groups[i],hop=j,reference_relation=r['new_triples_labeled'][j][1],changed=old!=new,category=category,doc_id=did,review='machine_reviewed_exact_triple_mapping',offline_only=True))
    with (out/'bridge_availability.jsonl').open('w') as f:
        for r in bridge:f.write(json.dumps(r)+'\n')
    chosen_qids={f'T_{i:06d}_q{v}' for i in chosen for v in range(3)}
    compare_ids=[min(i for i in split['split']['test'] if groups[i]==g) for g in split['groups']['test']]
    graph_qids=chosen_qids|{f'T_{i:06d}_q0' for i in compare_ids};generations=collections.defaultdict(list)
    for project in [parent,baseline]:
        p=project/'runs/initial_20260914/model_calls.jsonl'
        if not p.exists():continue
        with p.open() as f:
            for line in f:
                if 'Parse only the question into variables' not in line:continue
                r=json.loads(line);qid=r.get('context',{}).get('qid')
                if qid in graph_qids:
                    item={k:r.get(k) for k in ['key','text','prompt','finish','truncated','input_tokens','output_tokens']};item['source_log']=str(p)
                    if not any(x['key']==item['key'] for x in generations[qid]):generations[qid].append(item)
    ga=[];operator=[];identity_diffs=[];slot_counts=collections.Counter();activity_cache_misses=[]
    oldconfig=yaml.safe_load((baseline/'configs/frontier.yaml').read_text());selections=json.loads((parent/'manifests/adapt_qa_lock.json').read_text())['identity']['all_trained_selections']
    torch.set_num_threads(8);model=ReBindModule(d=128,layers=4).eval();model.load_state_dict(torch.load(selections['rebind_s17']['path'],weights_only=True,map_location='cpu')['model'])
    for qid in sorted(graph_qids):
        i=int(qid.split('_')[1]);paths=[baseline/'runs/frontier_dev'/(qid+'_open_native_s17.json'),baseline/'runs/frontier_test'/(qid+'_open_native_s17.json'),parent/'runs/mquake_T_open'/(qid+'_rebind.json')];p=next((p for p in paths if p.exists()),None)
        if p is None:ga.append(dict(qid=qid,status='no_historical_trace',raw_generations=generations[qid],review='pending'));continue
        r=json.loads(p.read_text());trace=next((t for t in r['trace'] if 'graph' in t),None)
        if trace is None:ga.append(dict(qid=qid,status='no_valid_graph',raw_generations=generations[qid],trace_path=str(p),review='machine_reviewed'));continue
        graph=trace['graph'];slot_counts[str(len(graph['slots']))]+=1
        parsed=[]
        for g in generations[qid]:
            try:
                obj=json.loads(g['text']);parsed.append(dict(key=g['key'],raw_slots=obj.get('slots'),raw_slot_count=len(obj.get('slots',[])),raw_variables=obj.get('variables'),truncated=g['truncated']))
            except (ValueError,TypeError):parsed.append(dict(key=g['key'],parse_failure=True))
        ga.append(dict(qid=qid,question=byqid[qid]['question'],group=groups[i],trace_path=str(p),raw_generations=generations[qid],parsed_generations=parsed,final_graph=graph,reference_relations=[x[1] for x in rows[i]['new_triples_labeled']],reference_hops=len(rows[i]['new_triples']),review='machine_reviewed_pipeline; semantic_equivalence_pending',note='Reference hop count is descriptive, not an automatic graph correctness label. No online access to this record.'))
        domain=trace['candidates'];n=len(domain);ks=[len(xs) for xs in domain.values()];legal=sum(ks[a]*ks[b]*max(0,ks[k]-1) for a in range(n) for b in range(n) for k in range(n) if len({a,b,k})==3)
        op=dict(qid=qid,round=trace['round'],variables=n,explicit_interpretation_graph_nodes=0,interpretation_special_categories_per_variable=2,valid_candidates=ks,unknown_fraction=n/sum(ks),legal_candidate_message_paths=legal,nonzero_messages=0 if legal==0 else None,actual_forward_checked=False)
        if i in compare_ids:
            proposal=trace['proposal'];key=digest(dict(feature_version='centered_window_v2',cache_context=dict(qid=qid,split='frontier_test'),question=byqid[qid]['question'],graph=graph,candidates=domain,docs=proposal['visible'],proposal=proposal,config=oldconfig));feature=baseline/'data/features'/(key+'.pt')
            if feature.exists():
                f=torch.load(feature,weights_only=True,map_location='cpu');messages=[]
                original=model.triangle.message
                def observed(*args,**kwargs):
                    result=original(*args,**kwargs);messages.append(dict(nonzero=int(torch.count_nonzero(result)),l1=float(result.abs().sum())));return result
                model.triangle.message=observed
                with torch.no_grad():full=model(**f)
                model.triangle.message=original
                valid=f['valid'];beam=decode(full['unary'],full['pair'],valid,beam=8)
                independent=decode(full['unary'],torch.zeros_like(full['pair']),valid,beam=8)
                op.update(actual_forward_checked=True,feature_path=str(feature),tensor_shape=list(f['candidate'].shape),allowed_shape=list(f['allowed'].shape),nonzero_messages=sum(m['nonzero'] for m in messages),layer_messages=messages,pair_potential_changes_top_decode=beam[0][1]!=independent[0][1],pair_lesion_scope='Zero final pair potential only; does not isolate trained triangle contribution.')
            else:activity_cache_misses.append(dict(qid=qid,path=str(feature)))
        operator.append(op)
    with (out/'graph_audit.jsonl').open('w') as f:
        for r in ga:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(out/'operator_activity.json',dict(records=operator,feature_cache_misses=activity_cache_misses,legal_mask_source='FeatureBuilder sets allowed to ones; TriangleUpdate requires distinct i,j,k and excludes UNKNOWN bridge. Source interpretation scores do not add graph nodes.',summary=dict(records=len(operator),zero_legal_paths=sum(r['legal_candidate_message_paths']==0 for r in operator),actual_forward_checked=sum(r['actual_forward_checked'] for r in operator))))
    for i in compare_ids:
        qid=f'T_{i:06d}_q0';a=parent/'runs/adapt_qa'/(qid+'_adapted_rebind_s17.json');b=baseline/'runs/frontier_test'/(qid+'_open_native_s17.json');ra=json.loads(a.read_text());rb=json.loads(b.read_text())
        signature=lambda r:{'queries':[t['query'] for t in r['trace']],'graphs':[t.get('graph') for t in r['trace']],'candidates':[t.get('candidates') for t in r['trace']],'visible_spans':[t.get('visible_spans') for t in r['trace']],'reader_prompt':r['final_prompt'],'answer':r['answer']}
        sa,sb=signature(ra),signature(rb);identity_diffs.append(dict(qid=qid,group=groups[i],old_path=str(a),new_path=str(b),same={k:digest(sa[k])==digest(sb[k]) for k in sa},old_hash=digest(a),new_hash=digest(b),graph_generation_keys=[x['key'] for x in generations[qid]]))
    write(out/'baseline_identity_diff.json',dict(records=identity_diffs,old_config=json.loads(json.dumps(yaml.safe_load((parent/'configs/adapt.yaml').read_text()))),new_config=oldconfig,explanation='Config/workdir/context cache identities differ and outputs were regenerated. This audit locates observable divergence, not a unique attribution to backend nondeterminism. New comparisons must rerun baselines concurrently.'))
    sourcefiles=[parent/'reports/MQUAKE_TRAINING.md',parent/'configs/adapt.yaml',parent/'configs/adapt_continue.yaml',parent/'src/rebind_mvp/adapt.py',parent/'reports/adapt_learning_curves.jsonl',parent/'manifests/adapt_qa_lock.json',root/'data/raw/T-00000-of-00001.parquet',root/'data/memory/T.json',root/'data/memory/T_e5.npy']+[pathlib.Path(x['path']) for name,x in selections.items() if name.startswith(('rebind_','bp_rebind_'))]
    modelpath=pathlib.Path(c['models']['generator_path']);sourcefiles+=list(modelpath.glob('*.safetensors'))+[modelpath/'config.json',modelpath/'tokenizer.json']
    retrieverpath=pathlib.Path(c['models']['retriever_path']);sourcefiles+=list(retrieverpath.glob('*.safetensors'))
    write(root/'manifests/sources.lock.json',dict(files={str(p):digest(p) for p in sourcefiles},base_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initial_dirty_diff=subprocess.check_output(['git','diff'],text=True),model_paths=dict(generator=str(modelpath),retriever=str(retrieverpath)),phase='before frontend repair'))
    summary=dict(dev_cases=len(chosen),dev_build_cases=len(manifest['dev_build']),dev_check_cases=len(manifest['dev_check']),dev_groups=len(devgroups),dev_build_groups=7,dev_check_groups=7,fresh_groups=0,memory_documents=len(memory),mapped_memory_documents=len(mapped),bridge_categories=dict(collections.Counter(r['category'] for r in bridge)),historical_graph_records=len(ga),historical_slot_counts=dict(slot_counts),historical_missing_traces=sum(r.get('status')=='no_historical_trace' for r in ga),operator_records=len(operator),actual_forward_checked=sum(r['actual_forward_checked'] for r in operator),training_loss='Both stages: .5*(unary CE + .25 pair CE + .25 interpretation CE), plus .5 revision and .1 stability. No direct frontier term in actual adapt.loss_and_metrics.',semantic_review='machine_reviewed/pending, no human_valid labels')
    write(out/'audit_summary.json',summary)
    (out/'audit_summary.md').write_text('# Offline audit\n\n'+json.dumps(summary,indent=2)+'\n\nAll reference labels remain offline. Single-slot counts are not semantic correctness scores. Raw generation, parsed slots and retained graph are in graph_audit.jsonl; no slots[0] truncation exists in the inspected proposer.\n')
    return summary


def online(c,phase,partition,tag):
    import os, numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from rebind_mvp.proposal import Generator
    from rebind_mvp.retrieval import E5
    from rebind_mvp.mquake import EditRetriever
    from rebind_mvp.schema import InferenceExample
    from rebind_mvp.transitions import NeuralState
    from rebind_mvp.evaluate import run_loop
    from rebind_mvp.final_frontend import run_repaired, GRAPH_PROMPT, validate_graph, next_action
    root=pathlib.Path(c['paths']['workdir']);parent=pathlib.Path(c['diagnostics']['parent']);manifest=json.loads((root/'manifests/split_manifest.json').read_text());c=copy.deepcopy(c)
    selections=json.loads((parent/'manifests/adapt_qa_lock.json').read_text())['identity']['all_trained_selections']
    if phase in ['dev-aligned','locked-eval']:
        proposed=json.loads((root/'manifests/aligned_candidates.json').read_text())['checkpoints'];selections.update({'aligned_'+k:v for k,v in proposed.items()})
    c['component_versions']=dict(parser=digest([GRAPH_PROMPT,inspect.getsource(validate_graph)]),frontend=digest(root/'src/rebind_mvp/final_frontend.py'),state=digest(root/'src/rebind_mvp/transitions.py'),query=digest(inspect.getsource(next_action)),reader=digest(c['models']['answer_prompt']),data=digest(root/'data/memory/T.json'),checkpoints={k:v['sha256'] for k,v in selections.items() if k.startswith(('bp_rebind_','rebind_','aligned_'))})
    ids=json.loads((root/'manifests/alignment_prefix_lock.json').read_text())['train_cases'] if phase=='align-train' else manifest['dev_'+partition]
    if phase=='repair-smoke':
        g={i:r['group'] for r in manifest['groups'] for i in r['cases']};ids=[min(i for i in ids if g[i]==group) for group in sorted({g[i] for i in ids})]
    if phase=='locked-eval':
        frozen=root/'manifests/final_evaluation_lock.json'
        if not frozen.exists():raise ValueError('No frozen final evaluation decision; finish development and any justified alignment first')
        settings=json.loads(frozen.read_text());ids=manifest['D_seen'];assert settings['components']==c['component_versions']
    public=[r for r in json.loads((root/'data/public/T.json').read_text()) if r['case_id'] in set(ids)]
    specs=[dict(name='common_native',mode='ircot_common'),dict(name='json_fix',mode='json_fix',knowledge='hybrid')]
    for seed in ([17,29,43] if phase in ['locked-eval','dev-aligned'] else [17]):
        for mode in ['bp_rebind','rebind']:
            checkpoint=settings['selected_checkpoints'][f'{mode}_s{seed}'] if phase=='locked-eval' else selections[f'{mode}_s{seed}']
            if phase=='locked-eval':
                assert digest(pathlib.Path(checkpoint['path']))==checkpoint['sha256'], 'Selected checkpoint changed after lock'
            specs.append(dict(name=f'{mode}_fix_s{seed}',mode=mode,knowledge='hybrid',seed=seed,checkpoint=checkpoint))
    if phase=='dev-aligned':
        for seed in [17,29,43]:
            for mode in ['bp_rebind','rebind']:specs.append(dict(name=f'{mode}_aligned_s{seed}',mode=mode,knowledge='hybrid',seed=seed,checkpoint=selections[f'aligned_{mode}_s{seed}']))
    if phase=='repair-smoke':specs.append(dict(name='rebind_strict_s17',mode='rebind',knowledge='strict',seed=17,checkpoint=selections['rebind_s17']))
    if phase=='align-train':specs=[s for s in specs if s['mode'] in ['json_fix','rebind']]
    stage=phase.replace('-','_')+'_'+partition+'_'+tag;folder=root/'runs/final'/stage;folder.mkdir(parents=True,exist_ok=True)
    identity=dict(stage=stage,config=c,rows_hash=digest(public),split_hash=digest(root/'manifests/split_manifest.json'),specs=specs,runner=digest(inspect.getsource(online)),sources={str(p.relative_to(root)):digest(p) for p in (root/'src/rebind_mvp').rglob('*.py')},memory_hash=digest(root/'data/memory/T.json'),vectors_hash=digest(root/'data/memory/T_e5.npy'))
    h=digest(identity);lock=root/'manifests'/(stage+'_lock.json')
    if lock.exists():assert json.loads(lock.read_text())['identity']==identity,'Stage identity changed; retain old run and choose an explicit new tag'
    else:write(lock,dict(identity=identity,hash=h,time=time.time()))
    (folder/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False));generator=Generator(c);encoder=E5(c['models']['retriever_path'],device='cuda:3');docs=json.loads((root/'data/memory/T.json').read_text());vectors=np.load(root/'data/memory/T_e5.npy')
    def task(row):
        results=[];example=InferenceExample(row['qid'],row['question'])
        for spec in specs:
            path=folder/(row['qid']+'_'+spec['name']+'.json')
            if path.exists():r=json.loads(path.read_text());assert r['identity_hash']==h;results.append(r);continue
            generator.context=dict(qid=example.qid,split=stage,components=c['component_versions'],stage='common' if spec['mode']=='ircot_common' else 'structured',updater=spec['mode'],checkpoint=spec.get('checkpoint',{}).get('sha256'))
            first=len(generator.calls);retriever=EditRetriever(docs,vectors,row['allowed_doc_ids'],encoder,c['retrieval']['top_k']);neural=None
            try:
                if spec['mode']=='ircot_common':r=run_loop(example,retriever,generator,c,'ircot_common')
                else:
                    if 'checkpoint' in spec:
                        neural=NeuralState(encoder,c,pathlib.Path(spec['checkpoint']['path']),spec['mode'],split=stage);neural.checkpoint_hash=spec['checkpoint']['sha256']
                    r=run_repaired(example,retriever,generator,c,spec['mode'],neural,knowledge=spec['knowledge'])
            except Exception as error:
                import traceback
                r=dict(answer='',eval_status='runtime_failure',trace=[],fallback=1,error=repr(error),traceback=traceback.format_exc())
            calls=generator.calls[first:];r.update(qid=row['qid'],case_id=row['case_id'],variant=row['variant'],method=spec['name'],seed=spec.get('seed'),identity_hash=h,input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls),actual_requests=sum(not x['cache_hit'] for x in calls),actual_retriever_calls=retriever.calls,checkpoint_hash=spec.get('checkpoint',{}).get('sha256'))
            write(path,r);results.append(r);print(stage,row['qid'],spec['name'],r['eval_status'],flush=True);del neural
        return results
    with ThreadPoolExecutor(max_workers=int(os.environ.get('REBIND_WORKERS','32'))) as pool:records=[r for batch in pool.map(task,public) for r in batch]
    summary=dict(stage=stage,records=len(records),expected=len(public)*len(specs),failures=sum(r['eval_status']!='ok' for r in records),frontend_failures=sum(bool(r.get('frontend_failure')) for r in records),hash=h)
    write(folder/'summary.json',summary);return summary


def collect_alignment(c,tag):
    root=pathlib.Path(c['paths']['workdir']);manifest=json.loads((root/'manifests/split_manifest.json').read_text());path=root/'manifests/alignment_prefix_lock.json'
    # Selection is fixed before collecting training trajectories and uses no QA correctness.
    selected=sorted(i for g in manifest['groups'] if g['original_split']=='train' for i in sorted(g['cases'],key=lambda i:digest([20260915,'alignment',i]))[:2])
    identity=dict(train_cases=selected,train_groups=sorted(g['group'] for g in manifest['groups'] if g['original_split']=='train'),per_group_limit=2,all_variants=True,policies=['json_fix','rebind_fix_s17'],shared_training_pool='Union of real deployed JSON and frozen ReBind prefixes; both trainable operators receive the same pool and masks',split_hash=digest(root/'manifests/split_manifest.json'),no_correctness_selection=True,scope='Training-prefix collection only; no optimizer updates',tag=tag)
    if path.exists():assert json.loads(path.read_text())==identity,'Alignment collection identity changed'
    else:write(path,identity)
    write(root/'reports/training_summary.json',dict(status='collecting_deployed_prefixes',optimizer_updates=0,trigger='Dev-build exposed 201 parametric-hypothesis candidate occurrences without source links, unlike controlled gold-path training. Domain adaptation will be evaluated once, with equal BP/ReBind prefixes; 267/270 reference-surface-available variable rounds already choose the matching surface, so gains are uncertain.',frontier_loss=False,frontier_reason='Shared scheduler does not use the frontier head',private_labels_in_online_calls=False,selection=identity))
    return online(c,'align-train','train',tag)


def report(c,stage):
    import csv, numpy as np
    from rebind_mvp.mquake import official_function
    root=pathlib.Path(c['paths']['workdir']);folder=root/'runs/final'/stage;out=root/'reports'/stage;out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((root/'manifests'/(stage+'_lock.json')).read_text());records=[json.loads(p.read_text()) for p in sorted(folder.glob('T_*.json'))];gold=json.loads((root/'data/private/T.json').read_text());check=official_function(root,'check_answer');manifest=json.loads((root/'manifests/split_manifest.json').read_text());groups={i:r['group'] for r in manifest['groups'] for i in r['cases']};public={r['qid']:r for r in json.loads((root/'data/public/T.json').read_text())}
    bymethod=collections.defaultdict(list);failures=[];coverage=collections.defaultdict(collections.Counter);predictions=[];reader_pairs=collections.defaultdict(list)
    for r in records:
        assert r['identity_hash']==lock['hash'];r['correct']=bool(check(True,gold[str(r['case_id'])],r['answer']));bymethod[r['method']].append(r)
        assert [t['query'] for t in r['trace']]==[t['query'] for t in r['actual_retriever_calls']], 'Recorded trajectory differs from actual retrieval calls'
        verified_spans=set()
        paired=r.get('reader_pair',{});pair_scores={k:bool(check(True,gold[str(r['case_id'])],v['answer'])) for k,v in paired.items()}
        if paired:reader_pairs[r['method']].append(dict(qid=r['qid'],group=groups[r['case_id']],raw=pair_scores['raw'],state=pair_scores['state']))
        predictions.append({k:r.get(k) for k in ['qid','case_id','variant','method','seed','answer','correct','eval_status','fallback','identity_hash']}|{'reader_pair_scores':pair_scores})
        if r['eval_status']!='ok' or r.get('fallback'):failures.append({k:r.get(k) for k in ['qid','method','eval_status','error','frontend_failure','fallback']}|{'policy_errors':[e for t in r['trace'] for e in t.get('policy_errors',[])]})
        counter=coverage[r['method']];counter['questions']+=1;counter['graph_failures']+=bool(r.get('frontend_failure'))
        if 'compile_audit' in r:
            g=r['compile_audit']['final_graph'];counter['compiled_questions']+=1;counter['total_slots']+=len(g['slots']);counter['multiple_slot_questions']+=len(g['slots'])>1
        for t in r['trace']:
            counter['retrieval_rounds']+=1
            if set(t.get('retrieved_ids',[]))-set(public[r['qid']]['allowed_doc_ids']):raise ValueError('Disallowed retrieval document')
            action=t.get('action')
            if action:
                counter['executed_structured_actions']+=1;counter['executed_'+action['slot_id']]+=1
                counter['unknown_input_actions']+=any(v=='UNKNOWN' for v in action['inputs'].values())
            for event in t.get('candidate_events',[]):counter[event['origin_kind']+'_events']+=1
            if 'compile_audit' in r:
                assert t['graph']==r['compile_audit']['final_graph'], 'Candidate update changed the question graph'
                verified_spans.update(p['span_id'] for p in t.get('proposal',{}).get('accepted',[]))
                for xs in t['candidates'].values():
                    for candidate in xs:
                        if candidate.get('origin_kind')=='parametric_hypothesis':assert not candidate['origin_span_ids'] and not candidate.get('source_doc_ids') and not candidate['verified']
                        if candidate.get('origin_kind')=='question_anchor':assert candidate['surface'] in public[r['qid']]['question']
                        if candidate.get('origin_kind')=='retrieved':assert set(candidate['origin_span_ids'])<=verified_spans and set(candidate['source_doc_ids'])<=set(public[r['qid']]['allowed_doc_ids'])
            for e in t.get('policy_errors',[]):counter['error_'+e['stage']]+=1
            activity=t.get('state',{}).get('operator_activity')
            if activity:
                counter['operator_measured_rounds']+=1;counter['legal_message_paths']+=activity['legal_message_paths'];counter['nonzero_message_entries']+=sum(activity['nonzero_messages_by_layer']);counter['rounds_with_nonzero_messages']+=any(activity['nonzero_messages_by_layer']);counter['pair_lesion_changed_decode']+=activity['pair_potential_changes_top_decode']
        if paired:
            assert r['state_prompt'].split('\nUncertain current structure (not evidence): ')[0]==r['final_prompt'];counter['reader_pairs']+=1;counter['state_repairs']+=pair_scores['state'] and not pair_scores['raw'];counter['state_harms']+=pair_scores['raw'] and not pair_scores['state']
    table=[];scores={};pairs=[]
    for method,rr in sorted(bymethod.items()):
        cases=collections.defaultdict(list)
        for r in rr:cases[r['case_id']].append(r)
        complete={i:rs for i,rs in cases.items() if sorted(x['variant'] for x in rs)==[0,1,2]};scores[method]={r['qid']:float(r['correct']) for r in rr}
        table.append(dict(method=method,questions=len(rr),cases=len(complete),groups=len({groups[r['case_id']] for r in rr}),question_accuracy=float(np.mean([r['correct'] for r in rr])),case_accuracy=float(np.mean([any(x['correct'] for x in rs) for rs in complete.values()])),case_0of3=sum(sum(x['correct'] for x in rs)==0 for rs in complete.values()),case_1of3=sum(sum(x['correct'] for x in rs)==1 for rs in complete.values()),case_2of3=sum(sum(x['correct'] for x in rs)==2 for rs in complete.values()),case_3of3=sum(sum(x['correct'] for x in rs)==3 for rs in complete.values()),failures=sum(r['eval_status']!='ok' for r in rr),frontend_failures=sum(bool(r.get('frontend_failure')) for r in rr),input_tokens=sum(r.get('input_tokens',0) for r in rr),retrieval_calls=sum(len(r['actual_retriever_calls']) for r in rr)))
    comparisons=[('rebind_fix_s17','common_native'),('rebind_fix_s17','json_fix'),('rebind_fix_s17','bp_rebind_fix_s17')]+[(f'{mode}_aligned_s{seed}',f'{mode}_fix_s{seed}') for mode in ['rebind','bp_rebind'] for seed in [17,29,43]]
    for a,b in comparisons:
        if a not in scores or b not in scores:continue
        gg=collections.defaultdict(list)
        for q in sorted(scores[a].keys()&scores[b].keys()):gg[groups[public[q]['case_id']]].append(scores[a][q]-scores[b][q])
        sizes=np.array([len(v) for v in gg.values()]);sums=np.array([sum(v) for v in gg.values()]);draw=np.random.default_rng(612).integers(0,len(gg),(2000,len(gg)));boot=sums[draw].sum(1)/sizes[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);pairs.append(dict(treatment=a,control=b,groups=len(gg),difference=float(sums.sum()/sizes.sum()),ci_low=float(lo),ci_high=float(hi),leave_one_group_out=[dict(omitted=g,difference=float((sums.sum()-sums[j])/(sizes.sum()-sizes[j]))) for j,g in enumerate(gg) if len(gg)>1]))
    for name,rs in [('results',table),('paired_results',pairs)]:
        with (out/(name+'.csv')).open('w') as f:
            if rs:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    for name,rs in [('predictions',predictions),('failure_log',failures)]:
        with (out/(name+'.jsonl')).open('w') as f:
            for r in rs:f.write(json.dumps(r)+'\n')
    reader_summary=[]
    for method,rr in sorted(reader_pairs.items()):
        gg=collections.defaultdict(list)
        for r in rr:gg[r['group']].append(int(r['state'])-int(r['raw']))
        sizes=np.array([len(v) for v in gg.values()]);sums=np.array([sum(v) for v in gg.values()]);draw=np.random.default_rng(612).integers(0,len(gg),(2000,len(gg)));boot=sums[draw].sum(1)/sizes[draw].sum(1)
        reader_summary.append(dict(method=method,paired_questions=len(rr),groups=len(gg),raw_accuracy=float(np.mean([r['raw'] for r in rr])),state_accuracy=float(np.mean([r['state'] for r in rr])),repairs=sum(r['state'] and not r['raw'] for r in rr),harms=sum(r['raw'] and not r['state'] for r in rr),ci=np.quantile(boot,[.025,.975]).tolist(),group_differences={g:sum(v)/len(v) for g,v in gg.items()},scope='Paired reader diagnostic; frontend fallback questions excluded here, retained in primary QA denominator'))
    write(out/'reader_pairs.json',reader_summary)
    write(out/'intervention_coverage.json',dict(coverage));expected=json.loads((folder/'summary.json').read_text())['expected'] if (folder/'summary.json').exists() else None
    checks=dict(sources=all(digest(root/p)==sha for p,sha in lock['identity']['sources'].items()),runner=digest(inspect.getsource(online))==lock['identity']['runner'],memory=digest(root/'data/memory/T.json')==lock['identity']['memory_hash'],vectors=digest(root/'data/memory/T_e5.npy')==lock['identity']['vectors_hash'],split=digest(root/'manifests/split_manifest.json')==lock['identity']['split_hash'],checkpoints=all(digest(pathlib.Path(s['checkpoint']['path']))==s['checkpoint']['sha256'] for s in lock['identity']['specs'] if 'checkpoint' in s))
    unchanged=all(checks.values())
    status=dict(stage=stage,records=len(records),expected=expected,status='complete' if expected==len(records) and unchanged else 'in_progress',inference_unchanged=unchanged,failures=sum(r['eval_status']!='ok' for r in records),graph_failures=sum(bool(r.get('frontend_failure')) for r in records),hash=lock['hash'])
    status['identity_checks']=checks;status['trajectory_provenance_checks']='passed';write(out/'completion_audit.json',status);return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','repair-smoke','dev','dev-aligned','align-train','locked-eval','report']);p.add_argument('--partition',choices=['build','check','train'],default='build');p.add_argument('--collect-prefixes',action='store_true');p.add_argument('--prepare-shard',type=int,choices=range(4));p.add_argument('--method',choices=['rebind','bp_rebind']);p.add_argument('--seed',type=int,choices=[17,29,43]);p.add_argument('--epochs',type=int,default=5);p.add_argument('--tag',default='v1');p.add_argument('--stage');args=p.parse_args();c=yaml.safe_load(pathlib.Path('configs/final.yaml').read_text());start=time.time();status='ok'
    try:
        if args.phase=='audit':result=audit(c)
        elif args.phase=='report':result=report(c,args.stage)
        elif args.phase=='align-train' and args.collect_prefixes:result=collect_alignment(c,args.tag)
        elif args.phase=='align-train':
            from rebind_mvp.final_alignment import prepare,train
            if args.prepare_shard is not None:result=prepare(c,args.prepare_shard);result={k:v for k,v in result.items() if k!='rows'}
            else:
                if args.method is None or args.seed is None:raise ValueError('Specify --method and --seed for optimizer training, or --prepare-shard for label preparation')
                result=train(c,args.method,args.seed,args.epochs)
        else:result=online(c,args.phase,args.partition,args.tag)
        print(json.dumps(result),flush=True)
    except BaseException:status='failed';raise
    finally:append(pathlib.Path(c['paths']['workdir'])/'runs/final/commands.jsonl',dict(argv=sys.argv,start=start,seconds=time.time()-start,status=status,code_hash=digest(pathlib.Path(__file__))))
