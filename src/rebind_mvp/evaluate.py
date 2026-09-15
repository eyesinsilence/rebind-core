"""Public-input-only retrieval/generation loop; private metrics are downstream."""
import json,pathlib,time,copy
from dataclasses import asdict
from .schema import InferenceExample
from .audit import digest,append
from .proposal import propose
from .frontier import choose_frontier

ANSWER_PROMPT='''Answer the question from the supplied raw documents. Return only a JSON object with independent fields {"answer":"short answer","citations":["document ID"]}. If evidence is insufficient, answer "UNKNOWN". Do not put explanation into answer.\n'''
JSON_STATE_PROMPT='''Maintain an editable binding state. Reread old and new sources. Explicitly reconsider earlier assumptions, retain mutually exclusive alternatives, revoke wrong role bindings, and backtrack to a new query when needed. Select only the supplied grounded candidates, or UNKNOWN. Do not turn inferred bindings into observed facts. Return {"assignments":[{"v0":"candidate surface or UNKNOWN"}],"observed_slots":["s0"],"observed_evidence":{"s0":{"doc_id":"ID","quote":"literal source quote"}},"revisions":[],"answer_ready":false}.\n'''

def run_loop(example,retriever,generator,config,method='ircot_common',neural=None,fixed=None,proposal_cache=None,query_schedule=None,force_rounds=None,query_policy=None,use_query_head=True):
    if method=='off': method='ircot_common'
    docs={}; evidence_seen={}; queries=[]; trace=[]; graph=None; registry=None; thoughts=[]; state={}; fallback=0; calls_start=len(generator.calls); started=time.time()
    query=example.question; budget=config['retrieval']['max_query_calls_including_initial']
    if query_schedule is not None:
        assert fixed is None and 0<len(query_schedule)<=budget
        force_rounds=len(query_schedule)
    if force_rounds is not None: assert 0<force_rounds<=budget
    for round_index in range(budget):
        if fixed is not None:
            if round_index>=len(fixed): break
            query=fixed[round_index]['query']; retrieved=fixed[round_index]['retrieved_documents']
        else:
            if query_schedule is not None: query=query_schedule[round_index]
            retrieved=retriever.search(query)
        queries.append(query)
        for d in retrieved: docs.setdefault(d['doc_id'],d)
        ordered=list({d['doc_id']:d for d in retrieved+list(docs.values())}.values())
        raw,visible=generator.pack(ordered); current=dict(round=round_index,query=query,retrieved_documents=retrieved,retrieved_ids=[d['doc_id'] for d in retrieved],all_retrieved_ids=list(docs),seen_ids=[d['doc_id'] for d in visible],prompt_visible_ids=[d['doc_id'] for d in visible],visible_spans=[dict(doc_id=d['doc_id'],offsets=d['offsets']) for d in visible])
        for d in visible:
            if d['doc_id'] not in evidence_seen or len(d['text'])>len(evidence_seen[d['doc_id']]['text']): evidence_seen[d['doc_id']]=d
        current['ever_seen_ids']=list(evidence_seen)
        current['raw_evidence_tokens']=len(generator.tokenizer.encode(raw,add_special_tokens=False))
        ready=False; next_query=None
        try:
            if method not in ['ircot_common','direct_reader']:
                if proposal_cache is None:
                    graph,registry,proposal=propose(generator,example,list(evidence_seen.values()),graph,registry,round_index)
                else:
                    if round_index not in proposal_cache:
                        try:proposal_cache[round_index]=dict(result=copy.deepcopy(propose(generator,example,list(evidence_seen.values()),graph,registry,round_index)))
                        except (ValueError,KeyError,TypeError) as error:proposal_cache[round_index]=dict(error=repr(error))
                    shared=proposal_cache[round_index]
                    if 'error' in shared:raise ValueError('Shared proposal failure: '+shared['error'])
                    graph,registry,proposal=copy.deepcopy(shared['result'])
                current['proposal']=proposal; current['graph']=asdict(graph); current['candidates']=copy.deepcopy(registry.snapshot())
                if method=='json_rebind':
                    pending=list(evidence_seen.values())
                    while pending:
                        reread,block=generator.pack(pending)
                        compact_candidates={v:[dict(candidate_id=x['candidate_id'],surface=x['surface']) for x in candidates] for v,candidates in registry.snapshot().items()}
                        state_prompt=JSON_STATE_PROMPT+'Question: '+example.question+'\nGraph: '+json.dumps(asdict(graph))+'\nCandidates: '+json.dumps(compact_candidates)+'\nPrevious state: '+json.dumps(state)+'\nDocuments:\n'+reread
                        if config.get('mquake'):
                            domain={v:{'type':'string','enum':[x['surface'] for x in candidates]} for v,candidates in compact_candidates.items()}
                            schema={'type':'object','properties':{'assignments':{'type':'array','items':{'type':'object','properties':domain,'required':list(domain),'additionalProperties':False}}},'required':['assignments']}
                            proposed_state=generator.json(state_prompt,schema=schema)
                        else: proposed_state=generator.json(state_prompt)
                        if not isinstance(proposed_state.get('assignments',[]),list) or not all(isinstance(a,dict) and all(isinstance(v,str) for v in a.values()) for a in proposed_state.get('assignments',[])): raise ValueError('Invalid editable assignment schema')
                        proofs=proposed_state.get('observed_evidence',{})
                        if not isinstance(proofs,dict) or not all(isinstance(v,dict) for v in proofs.values()) or not isinstance(proposed_state.get('observed_slots',[]),list) or not all(isinstance(v,str) for v in proposed_state.get('observed_slots',[])): raise ValueError('Invalid observed evidence schema')
                        state=proposed_state
                        allowed_surfaces={v:{x['surface'] for x in candidates} for v,candidates in registry.snapshot().items()}
                        for assignment in state.get('assignments',[]):
                            for var,value in list(assignment.items()):
                                if value not in allowed_surfaces.get(var,set()): assignment[var]='UNKNOWN'
                        evidence_by_id={d['doc_id']:d for d in evidence_seen.values()}
                        proofs=state.get('observed_evidence',{})
                        state['observed_slots']=[slot for slot in state.get('observed_slots',[]) if slot in proofs and proofs[slot].get('doc_id') in evidence_by_id and proofs[slot].get('quote') and proofs[slot]['quote'] in evidence_by_id[proofs[slot]['doc_id']]['text']]
                        complete=len(block)
                        if len(block[-1]['text'])<len(pending[complete-1]['text']) and complete>1: complete-=1
                        pending=pending[complete:]
                else:
                    state=neural.update(example,graph,registry,proposal,proposal['visible'],round_index)
                current['state']=state; ready=bool(state.get('answer_ready',False))
                target_slot,next_query=choose_frontier(graph,state.get('query_assignments',state.get('assignments',[])) if use_query_head else state.get('assignments',[]),registry,queries,state.get('observed_slots',[]))
                if query_policy is not None and round_index<budget-1:
                    target_slot,next_query,policy_audit=query_policy(example,graph,state,queries,raw,round_index,target_slot,next_query)
                    current['query_policy']=policy_audit
                current['frontier_slot_id']=target_slot; current['frontier_query']=next_query
            if method=='ircot_common' or (not ready and next_query is None and round_index<budget-1):
                prompt='Read the question and documents. Produce JSON {"thought":"one grounded reasoning step","next_query":"specific next search","answer_ready":false}. Reconsider previous thoughts if unsupported.\nQuestion: '+example.question+'\nPrevious thoughts: '+json.dumps(thoughts)+'\nDocuments:\n'+raw
                current['reasoning_prompt']=prompt; reply=generator.json(prompt); current['reasoning_reply']=reply; thoughts.append(reply.get('thought','')); next_query=reply.get('next_query'); ready=bool(reply.get('answer_ready',False))
        except (ValueError,KeyError,TypeError) as e:
            fallback+=1; current['error']=repr(e)
            # One common IRCoT fallback for all structured parse/state errors.
            thought=generator.generate(['Generate one grounded next retrieval query. Question: '+example.question+'\nDocuments:\n'+raw],max_tokens=128)[0]
            next_query=thought.strip()
        trace.append(current)
        if fixed is not None: ready=round_index==len(fixed)-1
        if force_rounds is not None: ready=round_index==force_rounds-1
        if ready or round_index==budget-1 or (method=='direct_reader' and fixed is None): break
        if fixed is not None or query_schedule is not None: continue
        if not next_query or next_query in queries:
            next_query=example.question+' '+(visible[-1]['title'] if visible else '')
            current['fallback_query']='question_last_visible_title'
            if next_query in queries and force_rounds is None: break
        query=next_query
    raw,visible=generator.pack(ordered)
    prompt=config['models'].get('answer_prompt',ANSWER_PROMPT)+'Question: '+example.question+'\nDocuments:\n'+raw
    state_included=False; state_tokens=0; state_omitted_reason=None
    if state:
        reader_state={k:state[k] for k in ['assignments','status','observed_slots','revisions'] if k in state}
        if 'assignments' in reader_state: reader_state['assignments']=reader_state['assignments'][:2]
        if 'revisions' in reader_state: reader_state['revisions']=reader_state['revisions'][-4:]
        encoded=json.dumps(reader_state); tokens=generator.tokenizer.encode(encoded,add_special_tokens=False)
        state_tokens=len(tokens)
        if len(tokens)<=config['models']['structured_state_tokens']:
            prompt+='\nUncertain current structure (not evidence): '+encoded; state_included=True
        else: state_omitted_reason='structured_state_over_budget'; fallback+=1
    try:
        final=generator.json(prompt); answer=final['answer']; citations=final.get('citations',[])
        if not isinstance(answer,str) or not isinstance(citations,list): raise ValueError('Invalid answer fields')
        status='ok'
    except (ValueError,KeyError,TypeError) as e: answer=''; citations=[]; status='parse_failure'; fallback+=1
    citation_locatable=sum(cid in evidence_seen for cid in citations)
    actual=generator.calls[calls_start:]
    return dict(state_included=state_included,structured_state_tokens=state_tokens,state_omitted_reason=state_omitted_reason,qid=example.qid,method=method,answer=answer,citations=citations,citation_locatable=citation_locatable,citation_count=len(citations),eval_status=status,fallback=fallback,trace=trace,final_prompt=prompt,config_hash=digest(config),seconds=time.time()-started,query_count=len(queries),llm_calls=len(actual),llm_input_tokens=sum(c['input_tokens'] for c in actual),llm_output_tokens=sum(c['output_tokens'] for c in actual),cache_hits=sum(c['cache_hit'] for c in actual))
