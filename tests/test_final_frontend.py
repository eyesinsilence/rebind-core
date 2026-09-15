import copy
import pytest
from rebind_mvp.schema import CandidateRegistry
from rebind_mvp.final_frontend import validate_graph, supported_assignments, next_action, add_candidate
from rebind_mvp.joint_decoder import constraint_penalty


def graph(question='Who leads the country of X?'):
    return validate_graph(dict(variables=[dict(var_id='answer',is_answer=True,description='leader'),dict(var_id='x',anchor='X',description='person'),dict(var_id='c',description='country')],slots=[dict(slot_id='s1',ordered_arguments=['c','answer'],relation_text='head of state',dependencies=['s0']),dict(slot_id='s0',ordered_arguments=['x','c'],relation_text='citizenship',dependencies=[])],constraints=[]),question)


def test_full_array_and_rewrites_preserve_dependency_chain():
    for question in ['Who leads the country of X?','X belongs to which country, and who leads it?','Name the leader of the country where X has citizenship.']:
        g=graph(question)
        assert [s['slot_id'] for s in g.slots]==['s0','s1']
        assert g.slots[1]['dependencies']==['s0']
    obj=dict(variables=graph().variables,slots=graph().slots[1:],constraints=[])
    with pytest.raises(ValueError):validate_graph(obj,'Who leads the country of X?')


def test_unknown_and_unsupported_typed_candidates_cannot_bridge():
    g=graph();r=CandidateRegistry(g.variables)
    add_candidate(r,'x','X','question_anchor',None,{},0)
    r.add('c','Country Y','unrelated same-type country',[],0)
    state=supported_assignments(g,r,dict(assignments=[dict(x='X',c='Country Y',answer='UNKNOWN')]))
    assert state[0]['c']=='UNKNOWN'
    first=next_action(g,state,set());assert first['slot_id']=='s0'
    assert next_action(g,state,{first['key']}) is None


def test_hypothesis_provenance_and_new_parameters_reenable_query():
    g=graph();r=CandidateRegistry(g.variables);add_candidate(r,'x','X','question_anchor',None,{},0)
    cid=add_candidate(r,'c','Country Y','parametric_hypothesis','s0',dict(x='X'),0)
    assert r.pool['c'][cid]['origin_span_ids']==[] and not r.pool['c'][cid]['verified']
    before=copy.deepcopy(g)
    a=supported_assignments(g,r,dict(assignments=[dict(c='Country Y')]))
    first=next_action(g,a,set());second=next_action(g,a,{first['key']})
    assert second['slot_id']=='s1' and second['inputs']==dict(c='Country Y')
    add_candidate(r,'c','Country Z','parametric_hypothesis','s0',dict(x='X'),1)
    b=supported_assignments(g,r,dict(assignments=[dict(c='Country Z')]))
    changed=next_action(g,b,{first['key'],second['key']})
    assert changed['slot_id']=='s1' and changed['inputs']==dict(c='Country Z')
    assert g==before


def test_wrong_relation_input_cannot_gain_source_provenance():
    g=graph();r=CandidateRegistry(g.variables);doc=dict(doc_id='d',title='update',text='Country Y has leader Z.')
    with pytest.raises(ValueError):add_candidate(r,'c','Country Y','retrieved','s0',dict(x='X'),0,doc,'Country Y has leader Z.')
    with pytest.raises(ValueError):add_candidate(r,'c','Country Y','retrieved','s0',dict(x='X'),0,None,None)


def test_downstream_binding_is_invalidated_and_snapshot_is_immutable():
    g=graph();r=CandidateRegistry(g.variables);add_candidate(r,'x','X','question_anchor',None,{},0)
    add_candidate(r,'c','Country Y','parametric_hypothesis','s0',dict(x='X'),0)
    add_candidate(r,'c','Country Z','parametric_hypothesis','s0',dict(x='X'),1)
    doc=dict(doc_id='d',title='update',text='Country Y has leader A.')
    aid=add_candidate(r,'answer','A','retrieved','s1',dict(c='Country Y'),0,doc,doc['text'])
    snapshot=r.snapshot();r.pool['answer'][aid]['origin_span_ids'].append('later')
    assert 'later' not in snapshot['answer'][1]['origin_span_ids']
    state=supported_assignments(g,r,dict(assignments=[dict(c='Country Z',answer='A')]))
    assert state[0]['answer']=='UNKNOWN'
    domain=r.snapshot();idx=[next(i for i,c in enumerate(domain[v]) if c['surface']==target) for v,target in [('answer','A'),('x','X'),('c','Country Z')]]
    assert constraint_penalty(g,domain,tuple(idx))==float('-inf')


def test_parametric_candidate_has_no_retrieved_feature_link(tmp_path):
    import torch, numpy as np
    from rebind_mvp.transitions import FeatureBuilder
    class Tokenizer:
        def encode(self,s,**kwargs):return list(range(len(s.split())))
        def __call__(self,s,**kwargs):return {'offset_mapping':[(0,len(s))]}
    class Encoder:
        tokenizer=Tokenizer()
        def encode(self,texts,kind='query',tokens=False):
            if tokens:return torch.ones(len(texts),3,768),torch.ones(len(texts),3,dtype=torch.bool)
            return np.ones((len(texts),768),dtype=np.float32)
    g=graph();r=CandidateRegistry(g.variables);add_candidate(r,'x','X','question_anchor',None,{},0);cid=add_candidate(r,'c','Country Y','parametric_hypothesis','s0',dict(x='X'),0)
    f=FeatureBuilder(Encoder(),{'paths':{'data_root':str(tmp_path)},'final_plan':{'enabled':True}}).build('q',g,r,[dict(doc_id='d',title='update',text='Country Y has leader A.',offsets=[0,23])],dict(accepted=[]))
    vi=list(r.pool).index('c');ci=next(i for i,x in enumerate(r.snapshot()['c']) if x['candidate_id']==cid)
    assert not f['links'][:,vi,ci].any()
    assert not f['valid'][list(r.pool).index('x'),0]


@pytest.mark.parametrize('mode',['rebind','bp_rebind'])
def test_fixed_anchor_mask_keeps_operator_finite_and_gradients_valid(mode):
    import torch
    from rebind_mvp.source_reader import ReBindModule
    torch.manual_seed(12)
    model=ReBindModule(input_dim=12,d=8,layers=3,mode=mode)
    valid=torch.tensor([[True,False],[False,True],[True,True]])
    links=torch.zeros(2,3,2,dtype=torch.bool);links[:,:,0]=True
    out=model(candidate=torch.randn(3,2,12),roles=torch.randn(3,12),tokens=torch.randn(2,4,12),token_mask=torch.ones(2,4,dtype=torch.bool),links=links,valid=valid,allowed=torch.ones(3,3,3,dtype=torch.bool))
    assert torch.isfinite(out['unary'][valid]).all() and torch.isfinite(out['pair']).all()
    loss=out['unary'][valid].sum()+out['pair'].sum();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_unused_declaration_removed_without_losing_slots():
    g=graph();g.variables.append(dict(var_id='unused',description='redundant declaration'))
    fixed=validate_graph(g.__dict__,'Who leads the country of X?')
    assert len(fixed.variables)==3 and len(fixed.slots)==2


def test_off_answer_relation_branch_is_not_accepted_as_complete_chain():
    g=graph();g.slots[1]['ordered_arguments']=['x','answer']
    with pytest.raises(ValueError,match='Extraneous relation'):validate_graph(g.__dict__,'Who leads the country of X?')


def test_repaired_loop_uses_last_retrieval_and_preserves_question_graph(tmp_path):
    from rebind_mvp.final_frontend import run_repaired
    from rebind_mvp.schema import InferenceExample
    class Generator:
        context={};calls=[]
        tokenizer=type('Tokenizer',(),{'encode':lambda self,s:list(s)})()
        def pack(self,docs):return '\n'.join(d['text'] for d in docs),docs
        def json(self,prompt,**kwargs):
            if prompt.startswith('Compile'):return copy.deepcopy(graph().__dict__)
            if prompt.startswith('Extract'):return {'candidates':[]}
            if prompt.startswith('Propose'):return {'values':['Country Y']}
            if prompt.startswith('Select'):return {'assignments':[{'x':'X','c':'Country Y','answer':'UNKNOWN'}]}
            assert 'LAST RETRIEVAL' in prompt
            return {'answer':'A'}
    class Retriever:
        calls=[]
        def search(self,query):
            self.calls.append(query);text='Initial evidence' if len(self.calls)==1 else 'LAST RETRIEVAL'
            return [dict(doc_id=str(len(self.calls)),title='update',text=text,offsets=[0,len(text)])]
    config={'paths':{'data_root':str(tmp_path)},'final_plan':{'parser_tokens':512},'retrieval':{'max_query_calls_including_initial':2},'models':{'answer_prompt':'Answer: ','structured_state_tokens':1024},'component_versions':{'parser':'p','proposal':'c','state':'s','query':'q','reader':'r','data':'d','checkpoints':'m'}}
    retriever=Retriever();result=run_repaired(InferenceExample('public-q','Who leads the country of X?'),retriever,Generator(),config,'json_fix')
    assert len(retriever.calls)==2 and 'LAST RETRIEVAL' in result['final_prompt']
    assert result['trace'][0]['graph']==result['trace'][1]['graph']==graph().__dict__
    assert result['state_prompt'].startswith(result['final_prompt'])
    # The inference example supplies only qid/question. There is no offline label object.
    assert not any(event['origin_span_ids'] for t in result['trace'] for event in t['candidate_events'] if event['origin_kind']=='parametric_hypothesis')


def test_compiler_cache_identity_includes_every_component(tmp_path):
    from rebind_mvp.final_frontend import compile_graph
    from rebind_mvp.schema import InferenceExample
    class Generator:
        calls=[]
        def json(self,*args,**kwargs):return copy.deepcopy(graph().__dict__)
    config={'paths':{'data_root':str(tmp_path)},'final_plan':{'parser_tokens':512},'component_versions':{k:'v1' for k in ['parser','proposal','state','query','reader','data','checkpoints']}}
    example=InferenceExample('q','Who leads the country of X?');_,first=compile_graph(Generator(),example,config)
    for component in config['component_versions']:
        changed=copy.deepcopy(config);changed['component_versions'][component]='v2';_,new=compile_graph(Generator(),example,changed)
        assert first['cache_key']!=new['cache_key']


def test_alignment_collection_uses_training_groups_without_outcomes(tmp_path,monkeypatch):
    import importlib.util,json,pathlib
    spec=importlib.util.spec_from_file_location('final_runner',pathlib.Path(__file__).resolve().parents[1]/'scripts/run_final_plan.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    (tmp_path/'manifests').mkdir();(tmp_path/'reports').mkdir()
    manifest={'groups':[{'original_split':'train','group':'a','cases':[1,2,3]},{'original_split':'train','group':'b','cases':[4]},{'original_split':'dev','group':'c','cases':[999]}]}
    (tmp_path/'manifests/split_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(runner,'online',lambda c,phase,partition,tag:(phase,partition,tag))
    assert runner.collect_alignment({'paths':{'workdir':str(tmp_path)}},'v4')==('align-train','train','v4')
    selection=json.loads((tmp_path/'manifests/alignment_prefix_lock.json').read_text())
    assert len(selection['train_cases'])==3 and set(selection['train_cases'])<={1,2,3,4}
    assert selection['all_variants'] and selection['no_correctness_selection']
    assert json.loads((tmp_path/'reports/training_summary.json').read_text())['optimizer_updates']==0
