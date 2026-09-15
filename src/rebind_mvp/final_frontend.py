"""Shared question compiler, provenance-aware candidate channel and dependency scheduler.

This module never reads private answers, reference hops, split membership or audit labels.
"""
import copy, json, pathlib, re, time
from dataclasses import asdict
from .audit import digest, write
from .schema import QuestionGraph, CandidateRegistry
from .joint_decoder import constraint_penalty
from .evaluate import run_loop, JSON_STATE_PROMPT

GRAPH_PROMPT='''Compile ONLY the question into a complete executable directed relation graph. Do not solve any relation or invent an entity. Every nested relation that must be solved must be a separate slot, never hidden in a variable description. Include literal question entities as anchored variables. Keep time, role and scope restrictions. Direct questions may have one slot; nested questions need the actual full path. All edges go from known input arguments to one unknown output (LAST argument). Each output is produced by only one slot. Use the existing schema:
{"variables":[{"var_id":"answer","type":"entity","description":"requested role","is_answer":true},{"var_id":"x0","type":"entity","description":"literal question entity","is_answer":false,"anchor":"literal substring"},{"var_id":"v0","type":"entity","description":"intermediate role","is_answer":false}],"slots":[{"slot_id":"s0","ordered_arguments":["x0","v0"],"relation_text":"first required relation","dependencies":[],"order":0,"query_template":"first required relation of {x0}"},{"slot_id":"s1","ordered_arguments":["v0","answer"],"relation_text":"second required relation","dependencies":["s0"],"order":1,"query_template":"second required relation of {v0}"}],"constraints":[]}
This is an anonymous schema, not evidence. Preserve the actual question's necessary relations, directions and restrictions. The answer variable must be FIRST and the final relation must output answer, not a duplicate variable for the requested role. Use at most 7 variables and 6 slots, short descriptions and no explanations. Anchors must be literal entity names, not descriptions of an unknown entity or a possessive event phrase. A relation such as a person's citizenship or place of death must be an edge from the named person to an unknown place. Every slot must contribute to the answer's dependency path; omit unused declarations and unrelated branches. Unanchored variables must not contain proposed answers. Return JSON only. Question: '''


def validate_graph(obj,question):
    graph=QuestionGraph(**{k:obj.get(k,[]) for k in ['variables','slots','constraints']})
    variables=graph.variables;ids=[v['var_id'] for v in variables]
    if not variables or len(ids)!=len(set(ids)) or len(ids)>7:raise ValueError('Invalid variable list')
    answers=[v['var_id'] for v in variables if v.get('is_answer')]
    if len(answers)!=1 or ids[0]!=answers[0]:raise ValueError('Exactly one answer variable must be first')
    anchors={v['var_id'] for v in variables if v.get('anchor')}
    if not anchors or any(v.get('anchor') and v['anchor'] not in question for v in variables):raise ValueError('Missing or nonliteral question anchor')
    slots=graph.slots;sid=[s['slot_id'] for s in slots]
    if not slots or len(slots)>6 or len(set(sid))!=len(sid):raise ValueError('Invalid slots')
    producer={}
    for s in slots:
        args=s['ordered_arguments']
        if len(args)<2 or any(v not in ids for v in args) or not s.get('relation_text'):raise ValueError('Invalid relation arguments')
        if args[-1] in anchors or args[-1] in producer:raise ValueError('Output must have one producer and cannot overwrite an anchor')
        producer[args[-1]]=s['slot_id']
    reached=set(anchors);ordered=[];pending=slots[:]
    while pending:
        ready=[s for s in pending if set(s['ordered_arguments'][:-1])<=reached]
        if not ready:raise ValueError('Disconnected or cyclic path: a required intermediate relation is missing')
        for s in sorted(ready,key=lambda s:s['slot_id']):
            deps=[producer[v] for v in s['ordered_arguments'][:-1] if v in producer]
            if any(x not in sid for x in s.get('dependencies',[])):raise ValueError('Unknown dependency slot')
            s['dependencies']=deps;s['order']=len(ordered);ordered.append(s);pending.remove(s);reached.add(s['ordered_arguments'][-1])
    if answers[0] not in reached:raise ValueError('The final required relation MUST output the answer variable; it is currently disconnected')
    needed={answers[0]}
    for s in reversed(ordered):
        if s['ordered_arguments'][-1] not in needed:raise ValueError('Extraneous relation outside the answer dependency path; recompile the actual nested question')
        needed.update(s['ordered_arguments'][:-1])
    # Remove only unused declarations, never relation slots or intermediate arguments.
    graph.variables=[v for v in variables if v['var_id'] in reached]
    graph.slots=ordered
    return graph


def compile_graph(generator,example,config):
    key=digest(dict(question=example.question,prompt=GRAPH_PROMPT,config=config,component='question_compiler_v1'))
    path=pathlib.Path(config['paths']['data_root'])/'compiled_graphs'/(key+'.json')
    if path.exists():saved=json.loads(path.read_text());return validate_graph(saved['parsed'],example.question),saved
    attempts=[]
    for attempt in range(2):
        prompt=GRAPH_PROMPT+example.question
        if attempts:prompt+='\nPrevious compilation failed validation: '+attempts[-1]['error']+'\nRecompile the question; do not invent missing entities or answers.'
        try:
            obj=generator.json(prompt,max_tokens=config['final_plan']['parser_tokens'])
            call=generator.calls[-1] if generator.calls else {}
            attempts.append(dict(stage='draft',raw_text=call.get('text'),parsed=copy.deepcopy(obj),finish=call.get('finish'),truncated=call.get('truncated',False)))
            review='Compile a corrected executable graph after checking this draft against ONLY the question. Keep its schema. Check every named entity anchor is the entity name itself, never a phrase describing an unknown entity such as its developer or country. Every required nested relation must be an edge; inputs precede the unknown output. The final requested entity is the answer variable: do not add a name-of edge or an artificial city/type conversion edge just because the question asks for a name or city. Preserve real geographic relations if the question actually requires them. Remove unused variable declarations, but preserve all required relationships. All slots must feed the answer. Do not solve the question or supply candidate values. Return ONLY the full corrected graph JSON.\nQuestion: '+example.question+'\nDraft: '+json.dumps(obj)
            obj=generator.json(review,max_tokens=config['final_plan']['parser_tokens']);raw=copy.deepcopy(obj);graph=validate_graph(copy.deepcopy(obj),example.question)
            call=generator.calls[-1] if generator.calls else {}
            attempts.append(dict(stage='question_only_self_check',raw_text=call.get('text'),parsed=raw,finish=call.get('finish'),truncated=call.get('truncated',False)))
            saved=dict(question=example.question,parsed=raw,final_graph=asdict(graph),attempts=attempts,cache_key=key,review='machine_reviewed; semantic_completeness_pending')
            write(path,saved);return graph,saved
        except (ValueError,KeyError,TypeError) as error:attempts.append(dict(error=repr(error),raw=copy.deepcopy(generator.calls[-1]) if generator.calls else None))
    write(path.with_suffix('.failed.json'),dict(attempts=attempts));raise ValueError('Question compilation failed after two attempts: '+str(attempts))


def supported_assignments(graph,registry,state):
    domain=registry.snapshot();anchors={v['var_id']:v['anchor'] for v in graph.variables if v.get('anchor')};out=[]
    for assignment in state.get('assignments',[]) or [anchors]:
        current=dict(anchors)
        for slot in graph.slots:
            inputs={v:current.get(v,'UNKNOWN') for v in slot['ordered_arguments'][:-1]};v=slot['ordered_arguments'][-1];value=assignment.get(v,'UNKNOWN')
            matching=[x for x in domain[v] if x['surface']==value and x.get('slot_id')==slot['slot_id'] and x.get('input_values')==inputs]
            current[v]=value if all(x!='UNKNOWN' for x in inputs.values()) and matching else 'UNKNOWN'
        if current not in out:out.append(current)
    return out


FRONTEND_VERSION = 'action_bound_literal_provenance_v2'


def relation_task(slot, inputs, evidence_version=None):
    """Separate a bounded retrieval identity from evidence-specific extraction."""
    inputs = copy.deepcopy(inputs)
    query_key = digest(dict(slot=slot['slot_id'], inputs=inputs, scope=slot.get('scope')))
    key = digest(dict(query_key=query_key, evidence_version=evidence_version))
    return dict(slot_id=slot['slot_id'], inputs=inputs, input_values=copy.deepcopy(inputs),
                scope=copy.deepcopy(slot.get('scope')), evidence_version=evidence_version,
                query_key=query_key, key=key)


def visible_evidence_version(visible):
    # Packing/truncation changes are evidence changes; retrieval ordering alone is not.
    identities = [dict(doc_id=d['doc_id'], title=d.get('title', ''), text=d['text'],
                       offsets=d.get('offsets', [0, len(d['text'])])) for d in visible]
    return digest(sorted(identities, key=digest))


def extraction_tasks(graph, assignments, action, evidence_version):
    """The queried branch comes first, even if it is absent from current top-1."""
    tasks = []
    seen = set()
    slots = {s['slot_id']: s for s in graph.slots}

    def add(slot, inputs, for_action=False):
        if any(value == 'UNKNOWN' for value in inputs.values()):
            return
        task = relation_task(slot, inputs, evidence_version)
        if task['key'] not in seen:
            seen.add(task['key'])
            tasks.append(dict(task, for_action=for_action))

    if action is not None:
        add(slots[action['slot_id']], action['input_values'], True)
    for slot in graph.slots:
        for binding in assignments:
            add(slot, {v: binding.get(v, 'UNKNOWN') for v in slot['ordered_arguments'][:-1]})
    return tasks


def next_action(graph,assignments,executed,evidence_version=None):
    for slot in graph.slots:
        for binding in assignments:
            inputs={v:binding.get(v,'UNKNOWN') for v in slot['ordered_arguments'][:-1]}
            if any(x=='UNKNOWN' for x in inputs.values()):continue
            task=relation_task(slot,inputs,evidence_version)
            if task['query_key'] in executed or task['key'] in executed:continue
            template=slot.get('query_template','');fields=re.findall(r'\{([^{}]+)\}',template)
            query=template.format(**inputs) if fields and set(fields)==set(inputs) else slot['relation_text']+' '+' '.join(inputs.values())
            return dict(task,query=query)
    return None


def add_candidate(registry,var,surface,kind,slot,inputs,round_index,doc=None,quote=None,relation=None):
    origins=[];sourceids=[];provenance=[]
    if not isinstance(surface,str) or not surface.strip() or surface=='UNKNOWN':
        raise ValueError('Candidate must be a nonempty, named value')
    if kind=='retrieved':
        if not doc or not isinstance(quote,str) or not quote or quote not in doc['text'] or surface not in quote:
            raise ValueError('Candidate output must occur inside the cited literal quote')
        context=doc.get('title','')+' '+quote
        if any(not isinstance(x,str) or x=='UNKNOWN' or x.casefold() not in context.casefold() for x in inputs.values()):
            raise ValueError('Input arguments must occur in the cited quote or its title')
        start=doc.get('offsets',[0])[0]+doc['text'].index(quote)
        span_id=digest([doc['doc_id'],start,quote])
        origins=[span_id];sourceids=[doc['doc_id']]
        provenance=[dict(span_id=span_id,doc_id=doc['doc_id'],char_start=start,char_end=start+len(quote),
                         quote=quote,title_context=doc.get('title',''),verification_level='literal_provenance')]
    elif kind not in ['question_anchor','parametric_hypothesis']:raise ValueError('Unknown provenance kind')
    identity=[kind,slot,inputs,surface,sourceids];cid=registry.add(var,surface,identity,origins,round_index)
    candidate=registry.pool[var][cid]
    records={p['span_id']:p for p in candidate.get('provenance_records',[])}
    records.update({p['span_id']:p for p in provenance})
    candidate.update(origin_kind=kind,source_doc_ids=sourceids,slot_id=slot,input_values=copy.deepcopy(inputs),
                     literal_provenance=kind=='retrieved',relation_checked=False,
                     verification_level={'retrieved':'literal_provenance','question_anchor':'question_anchor',
                                         'parametric_hypothesis':'unverified_hypothesis'}[kind],
                     verified=kind=='question_anchor',provenance_records=list(records.values()),
                     relation_claim=dict(slot_id=slot,input_values=copy.deepcopy(inputs),output_value=surface,
                                         relation=copy.deepcopy(relation),status='not_checked'))
    return cid


def run_repaired(example,retriever,generator,config,method,neural=None,knowledge='hybrid'):
    started=time.time();start_calls=len(generator.calls);context=generator.context.copy();components=dict(config.get('component_versions',{}));trace=[];errors=[];docs={};executed=set();hypotheses=set();raw='';graph=None;state={};queries=[]
    components['frontend_execution']=FRONTEND_VERSION
    extraction_attempts={}
    def stage(name):generator.context={**context,'stage':name,'components':components,'updater':method if name in ['state','reader'] else 'shared','checkpoint':getattr(neural,'checkpoint_hash',None) if name in ['state','reader'] else None}
    stage('parser')
    try:graph,compile_audit=compile_graph(generator,example,config)
    except (ValueError,KeyError,TypeError) as error:
        stage('fallback');r=run_loop(example,retriever,generator,config,'ircot_common');r.update(frontend_failure=repr(error),fallback=r.get('fallback',0)+1);return r
    registry=CandidateRegistry(graph.variables,limit=6)
    for v in graph.variables:
        if v.get('anchor'):add_candidate(registry,v['var_id'],v['anchor'],'question_anchor',None,{},0)
    state={'assignments':[{v['var_id']:v.get('anchor','UNKNOWN') for v in graph.variables}]};query=example.question;action=None
    for round_index in range(config['retrieval']['max_query_calls_including_initial']):
        retrieved=retriever.search(query);queries.append(query)
        for d in retrieved:docs[d['doc_id']]=d
        ordered=list({d['doc_id']:d for d in retrieved+list(docs.values())}.values());raw,visible=generator.pack(ordered);record=dict(round=round_index,query=query,retrieved_documents=retrieved,retrieved_ids=[d['doc_id'] for d in retrieved],visible_spans=[dict(doc_id=d['doc_id'],offsets=d['offsets']) for d in visible],action=action,graph=copy.deepcopy(asdict(graph)),candidate_events=[],policy_errors=[])
        evidence_version=visible_evidence_version(visible)
        record.update(evidence_version=evidence_version,frontend_version=FRONTEND_VERSION,extraction_tasks=[],
                      retrieval_attempt=dict(status='completed',query=query,
                                             action_key=action['key'] if action else None,
                                             query_key=action['query_key'] if action else None))
        if action:executed.add(action['query_key'])
        bindings=supported_assignments(graph,registry,state);accepted=[];rejected=[]
        for execution in extraction_tasks(graph,bindings,action,evidence_version):
            slot=next(s for s in graph.slots if s['slot_id']==execution['slot_id'])
            inputs=execution['input_values'];var=slot['ordered_arguments'][-1]
            if execution['key'] in extraction_attempts:
                previous=extraction_attempts[execution['key']]
                record['extraction_tasks'].append(dict(execution,attempted=False,status='reused_attempt',
                    completed=previous['completed'],produced_candidates=False,
                    previous_status=previous['status'],previous_candidate_ids=previous['candidate_ids']))
                # Keep the same quote-centered feature inputs when the visible evidence is unchanged.
                accepted.extend(copy.deepcopy(previous['accepted']))
                continue
            task=dict(relation=slot['relation_text'],known_inputs=inputs,scope=slot.get('scope'),requested_role=next(v['description'] for v in graph.variables if v['var_id']==var));found=[]
            task_record=dict(execution,attempted=True,completed=False,status='failed',produced_candidates=False)
            accepted_start=len(accepted);rejected_start=len(rejected)
            stage('proposal')
            try:
                prompt='Extract up to two answers to ONLY this relation from the raw updates. A document must state the requested relationship for the supplied input entity, not merely mention a same-type entity. If none, return an empty candidates array. Return JSON {"candidates":[{"surface":"literal output value","doc_id":"ID","quote":"exact source substring"}]}.\nRelation task: '+json.dumps(task)+'\nDocuments:\n'+raw
                reply=generator.json(prompt,max_tokens=384)
                if not isinstance(reply,dict) or not isinstance(reply.get('candidates'),list):
                    raise ValueError('Extraction response requires a candidates array')
                for item in reply['candidates'][:2]:
                    try:
                        if not isinstance(item,dict):raise ValueError('Candidate must be an object')
                        doc=next((d for d in visible if d['doc_id']==item.get('doc_id') and isinstance(item.get('quote'),str) and item['quote'] in d['text']),None)
                        cid=add_candidate(registry,var,item['surface'],'retrieved',slot['slot_id'],inputs,round_index,doc,item.get('quote'),relation=slot);found.append(cid)
                        candidate=registry.pool[var][cid];quote=item['quote'];start=doc['offsets'][0]+doc['text'].index(quote)
                        accepted.append(dict(var_id=var,surface=item['surface'],doc_id=doc['doc_id'],quote=quote,
                            span_id=digest([doc['doc_id'],start,quote]),candidate_id=cid,char_start=start,char_end=start+len(quote),
                            slot_id=slot['slot_id'],input_values=copy.deepcopy(inputs),task_key=execution['key'],
                            evidence_version=evidence_version,verification_level=candidate['verification_level'],relation_checked=False))
                    except (ValueError,KeyError,TypeError) as error:rejected.append(dict(item=item,task_key=execution['key'],error=repr(error)))
                task_record.update(completed=True,status='completed')
            except (ValueError,KeyError,TypeError) as error:
                task_record['error']=repr(error)
                record['policy_errors'].append(dict(stage='proposal',slot=slot['slot_id'],input_values=copy.deepcopy(inputs),task_key=execution['key'],error=repr(error)))
            task_record.update(produced_candidates=bool(found),candidate_ids=found[:],rejected_candidates=len(rejected)-rejected_start)
            record['extraction_tasks'].append(task_record)
            extraction_attempts[execution['key']]=dict(task_record,accepted=copy.deepcopy(accepted[accepted_start:]))
            hkey=digest([slot['slot_id'],inputs]);is_answer=next(v.get('is_answer',False) for v in graph.variables if v['var_id']==var)
            if knowledge=='hybrid' and not found and not is_answer and hkey not in hypotheses:
                hypotheses.add(hkey);stage('hypothesis')
                try:
                    prompt='Propose at most two tentative values from your background knowledge for ONLY the supplied single relation and known inputs. Use canonical full entity names. Do not solve the downstream full question, do not fabricate citations, and use UNKNOWN if unsure. Supplied updates override remembered facts. Return JSON {"values":["value"]}.\nRelation task: '+json.dumps(task)+'\nQuestion (context only): '+example.question+'\nCurrently exposed updates:\n'+raw
                    reply=generator.json(prompt,max_tokens=128)
                    for surface in reply.get('values',[])[:2]:
                        if isinstance(surface,str) and surface.strip() and surface!='UNKNOWN':
                            cid=add_candidate(registry,var,surface,'parametric_hypothesis',slot['slot_id'],inputs,round_index);record['candidate_events'].append(dict(candidate_id=cid,origin_kind='parametric_hypothesis',slot=slot['slot_id'],inputs=inputs,surface=surface,origin_span_ids=[]))
                except (ValueError,KeyError,TypeError) as error:record['policy_errors'].append(dict(stage='hypothesis',slot=slot['slot_id'],error=repr(error)))
            record['candidate_events'] += [dict(candidate_id=cid,origin_kind='retrieved',slot=slot['slot_id'],inputs=inputs) for cid in found]
        before=copy.deepcopy(state);stage('state')
        try:
            if method=='json_fix':
                domain=registry.snapshot();compact={v:[{k:x.get(k) for k in ['surface','origin_kind','slot_id','input_values','source_doc_ids','verification_level','relation_checked']} for x in xs] for v,xs in domain.items()}
                prompt=JSON_STATE_PROMPT+'Keep anchors fixed. A parametric_hypothesis is tentative knowledge, never source evidence. literal_provenance only checks quoted text; it does not verify a relationship, its direction or scope. A downstream candidate is usable only when its input_values match the current upstream bindings. Prefer a directly stated update over conflicting remembered hypotheses.\nQuestion: '+example.question+'\nGraph: '+json.dumps(asdict(graph))+'\nCandidates: '+json.dumps(compact)+'\nPrevious state: '+json.dumps(state)+'\nDocuments:\n'+raw
                schema={'type':'object','properties':{'assignments':{'type':'array','minItems':1,'maxItems':2,'items':{'type':'object','properties':{v:{'type':'string','enum':[x['surface'] for x in xs]} for v,xs in domain.items()},'required':list(domain),'additionalProperties':False}}},'required':['assignments']}
                prompt=prompt.replace(JSON_STATE_PROMPT,'Select at most two consistent joint bindings. Output ONLY {"assignments":[{"variable_id":"candidate surface"}]}; no explanations or other fields. ')
                proposed=generator.json(prompt,max_tokens=512,schema=schema);state=dict(assignments=supported_assignments(graph,registry,proposed),status='inferred_or_unresolved')
            else:state=neural.update(example,graph,registry,dict(accepted=accepted,rejected=rejected,visible=visible),visible,round_index)
        except (ValueError,KeyError,TypeError) as error:record['policy_errors'].append(dict(stage='state',error=repr(error)));state=before
        binding=supported_assignments(graph,registry,state);state['assignments']=binding
        if not binding:state['assignments']=supported_assignments(graph,registry,{})
        action=next_action(graph,state['assignments'],executed,evidence_version)
        record.update(candidates=registry.snapshot(),state=copy.deepcopy(state),next_action=action,proposal=dict(accepted=accepted,rejected=rejected,visible=visible))
        trace.append(record);errors+=record['policy_errors']
        if action is None:break
        query=action['query']
    stage('reader');base=config['models']['answer_prompt']+'Question: '+example.question+'\nDocuments:\n'+raw
    reader_state={k:state[k] for k in ['assignments','status','revisions'] if k in state};reader_state['assignments']=reader_state.get('assignments',[])[:2]
    with_state=base+'\nUncertain current structure (not evidence): '+json.dumps(reader_state)
    paired={}
    for mode,prompt in [('raw',base),('state',with_state)]:
        try:
            if mode=='state' and len(generator.tokenizer.encode(json.dumps(reader_state)))>config['models']['structured_state_tokens']:raise ValueError('Structured state exceeds separate state budget')
            final=generator.json(prompt);answer=final['answer'];assert isinstance(answer,str)
            paired[mode]=dict(answer=answer,eval_status='ok',prompt_hash=digest(prompt))
        except (ValueError,KeyError,TypeError,AssertionError) as error:paired[mode]=dict(answer='',eval_status='parse_failure',error=repr(error),prompt_hash=digest(prompt))
    calls=generator.calls[start_calls:]
    return dict(qid=example.qid,method=method,knowledge_mode=knowledge,answer=paired['raw']['answer'],eval_status=paired['raw']['eval_status'],reader_pair=paired,trace=trace,final_prompt=base,state_prompt=with_state,state_included=False,compile_audit=compile_audit,fallback=len(errors)+sum(r['eval_status']!='ok' for r in paired.values()),seconds=time.time()-started,input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls),llm_calls=len(calls),actual_requests=sum(not x['cache_hit'] for x in calls),max_prompt_tokens=max([x['input_tokens'] for x in calls] or [0]))
