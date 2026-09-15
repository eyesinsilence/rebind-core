import copy
import json
import threading
import types

import numpy as np
import pytest
import torch

from rebind_mvp.assignments import validated_state,check_relation
from rebind_mvp.final_frontend import add_candidate,validate_graph,next_action
from rebind_mvp.joint_decoder import constraint_penalty
from rebind_mvp.retrieval import E5
from rebind_mvp.runtime import generation_finish,model_revision,resolve_device
from rebind_mvp.schema import CandidateRegistry,InferenceExample,QuestionGraph
from rebind_mvp.transitions import FeatureBuilder,NeuralState


class Batch(dict):
    def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
    @property
    def input_ids(self):return self['input_ids']


class Tokenizer:
    def __call__(self,texts,**kwargs):
        if isinstance(texts,str):return {'offset_mapping':[(0,len(texts))]}
        lengths=[len(t.split()) for t in texts];width=max(lengths)
        mask=torch.tensor([[1]*n+[0]*(width-n) for n in lengths])
        return Batch(input_ids=mask.clone(),attention_mask=mask)
    def encode(self,text,**kwargs):return text.split()


def tiny_encoder(batch=2):
    encoder=E5.__new__(E5);encoder.local=threading.local();encoder.tokenizer_base=Tokenizer()
    encoder.embedding_dim=12;encoder.batch=batch;encoder.device='cpu'
    encoder.model=lambda **kw:types.SimpleNamespace(last_hidden_state=kw['input_ids'][...,None].float().expand(-1,-1,12))
    return encoder


def test_token_encoding_keeps_every_batch_with_different_padding():
    encoder=tiny_encoder();texts=['one','two words','three more words','four text tokens here','last']
    values,mask=encoder.encode(texts,tokens=True)
    assert values.shape==(5,5,12) and mask.sum(1).tolist()==[2,3,4,5,2]
    assert torch.equal(values[mask],torch.ones_like(values[mask]))
    assert not values[~mask].any()
    assert encoder.encode([]).shape==(0,12)
    assert encoder.encode([],tokens=True)[0].shape==(0,0,12)


@pytest.mark.parametrize('mode',['rebind','bp_rebind','frozen_old_read'])
def test_empty_retrieval_uses_neutral_padding_and_finite_real_model(tmp_path,mode):
    from rebind_mvp.source_reader import ReBindModule
    encoder=tiny_encoder();graph=QuestionGraph([dict(var_id='answer',description='answer')],[])
    registry=CandidateRegistry(graph.variables)
    config={'paths':{'data_root':str(tmp_path)},'rebind':{'hidden_dim':8,'layers':2,'joint_beam':8},'devices':{'scorer':'cpu'},'final_plan':{}}
    f=FeatureBuilder(encoder,config).build('q',graph,registry,[],{'accepted':[]})
    assert f['candidate'].shape[-1]==12 and f['source_doc_index'].tolist()==[-1]
    assert not f['links'].any() and f['token_mask'].all()
    path=tmp_path/'checkpoint.pt';torch.save({'model':ReBindModule(input_dim=12,d=8,layers=2).state_dict()},path)
    state=NeuralState(encoder,config,path,mode).update(InferenceExample('q','q'),graph,registry,{'accepted':[]},[],0)
    assert state['source_scores']=={} and state['candidate_assignments']


def test_empty_source_single_candidate_batched_forward_matches_single():
    from rebind_mvp.source_reader import ReBindModule
    from rebind_mvp.adapt import BatchedReBind
    single=ReBindModule(input_dim=12,d=8,layers=2)
    batched=BatchedReBind(input_dim=12,d=8,layers=2);batched.load_state_dict(single.state_dict())
    features=dict(candidate=torch.zeros(1,1,12),roles=torch.zeros(1,12),tokens=torch.zeros(1,1,12),
                  token_mask=torch.ones(1,1,dtype=torch.bool),links=torch.zeros(1,1,1,dtype=torch.bool),
                  valid=torch.ones(1,1,dtype=torch.bool),allowed=torch.ones(1,1,1,dtype=torch.bool))
    left=single(**features);right=batched(**{k:v[None] for k,v in features.items()})
    assert torch.allclose(left['unary'],right['unary'][0],atol=1e-6)
    assert torch.isfinite(right['pair']).all()


def identity_fixture():
    graph=validate_graph({'variables':[dict(var_id='answer',is_answer=True,description='leader'),dict(var_id='x',anchor='X',description='entity'),dict(var_id='place',description='place')],
                          'slots':[dict(slot_id='s0',ordered_arguments=['x','place'],relation_text='location'),dict(slot_id='s1',ordered_arguments=['place','answer'],relation_text='leader')],
                          'constraints':[]},'Who leads the location of X?')
    registry=CandidateRegistry(graph.variables)
    anchor=add_candidate(registry,'x','X','question_anchor',None,{},0)
    places=[]
    for did in ['france','texas']:
        doc=dict(doc_id=did,title='X',text='X is in Paris.')
        places.append(add_candidate(registry,'place','Paris','retrieved','s0',{'x':'X'},0,doc,doc['text'],input_candidate_ids={'x':anchor}))
    doc=dict(doc_id='leader',title='Paris',text='Paris has leader Alice.')
    answer=add_candidate(registry,'answer','Alice','retrieved','s1',{'place':'Paris'},0,doc,doc['text'],input_candidate_ids={'place':places[1]})
    return graph,registry,anchor,places,answer


def test_ambiguous_surface_does_not_select_an_arbitrary_identity():
    graph,registry,anchor,places,answer=identity_fixture()
    state=validated_state(graph,registry,{'assignments':[{'place':'Paris','answer':'Alice'}]})
    assert state['candidate_assignments'][0]['place']=='UNKNOWN'
    state=validated_state(graph,registry,{'candidate_assignments':[{'place':places[0],'answer':answer}]})
    assert state['candidate_assignments'][0]['place']==places[0] and state['assignments'][0]['answer']=='UNKNOWN'
    right=validated_state(graph,registry,{'candidate_assignments':[{'place':places[1],'answer':answer}]})
    assert right['assignments'][0]['answer']=='Alice'
    domain=registry.snapshot();ids={'x':anchor,'place':places[0],'answer':answer}
    indices=tuple(next(i for i,c in enumerate(domain[v]) if c['candidate_id']==ids[v]) for v in domain)
    assert constraint_penalty(graph,domain,indices)==float('-inf')


def test_same_name_actions_keep_distinct_candidate_identities():
    graph,registry,anchor,places,_=identity_fixture()
    surfaces=[{'x':'X','place':'Paris'}]*2
    identities=[{'x':anchor,'place':cid} for cid in places]
    first=next_action(graph,surfaces,set(),'v',identities)
    second=next_action(graph,surfaces,{first['query_key']},'v',identities)
    third=next_action(graph,surfaces,{first['query_key'],second['query_key']},'v',identities)
    assert second['query']==third['query'] and second['query_key']!=third['query_key']
    assert second['input_candidate_ids']!=third['input_candidate_ids']


def test_strict_admissibility_requires_positive_relation_verdict():
    graph,registry,anchor,places,answer=identity_fixture()
    proposed={'candidate_assignments':[{'place':places[1],'answer':answer}]}
    assert validated_state(graph,registry,proposed,'strict')['assignments'][0]['place']=='UNKNOWN'
    for var,cid in [('place',places[1]),('answer',answer)]:
        registry.pool[var][cid].update(relation_supported=True,relation_checked=True)
    assert validated_state(graph,registry,proposed,'strict')['assignments'][0]['answer']=='Alice'


@pytest.mark.parametrize('source',['assignments','candidate_assignments'])
@pytest.mark.parametrize('kind',['forbidden_assignment','required_role','allowed_types','equality','inequality','before','after'])
def test_json_and_neural_assignments_use_same_constraint_checker(source,kind):
    graph,registry,anchor,places,answer=identity_fixture()
    left=registry.pool['place'][places[1]];right=registry.pool['answer'][answer]
    if kind=='forbidden_assignment':condition={'type':kind,'candidate_ids':{'answer':answer}}
    elif kind=='required_role':
        right.update(role_verified=True,role='founder');condition={'type':kind,'arguments':['answer'],'role':'leader'}
    elif kind=='allowed_types':
        right['type']='person';condition={'type':kind,'arguments':['answer'],'types':['organization']}
    else:
        condition={'type':kind,'arguments':['place','answer']}
        left.update(entity_identity_verified=True,entity_id='L');right.update(entity_identity_verified=True,entity_id='R' if kind=='equality' else 'L')
        if kind in ['before','after']:
            left['surface']='2020';right['surface']='2010' if kind=='before' else '2030';right['input_values']={'place':'2020'}
    graph.constraints=[condition]
    row={'place':places[1],'answer':answer} if source=='candidate_assignments' else {'place':left['surface'],'answer':right['surface']}
    if source=='assignments':
        # Remove the unrelated homonym so this probe isolates constraints rather than ambiguity.
        registry.pool['place'].pop(places[0]);registry.active['place'].remove(places[0])
    state=validated_state(graph,registry,{source:[row]})
    assert state['assignments'][0]['answer']=='UNKNOWN'
    assert any(x['reason']=='graph_constraint_violation' for x in state['validation_issues'])


@pytest.mark.parametrize('verdict,support',[('supported',True),('unresolved',False),('contradicted',False)])
def test_relation_validation_requires_all_directed_claim_checks(verdict,support):
    reply=dict(verdict=verdict,inputs_match=True,output_matches=True,relation_matches=True,direction_matches=True,scope_matches=True)
    generator=types.SimpleNamespace(json=lambda *args,**kwargs:reply)
    slot=dict(relation_text='birthplace',ordered_arguments=['person','place'])
    result=check_relation(generator,slot,{'person':'Alice'},'Paris',{'title':'update'},'Alice attended a meeting. Bob was born in Paris.')
    assert result['relation_supported'] is support
    reply['direction_matches']=False
    assert not check_relation(generator,slot,{'person':'Alice'},'Paris',{'title':'update'},'quote')['relation_supported']


@pytest.mark.parametrize('reason,count,expected',[('stop',10,False),('length',2,True),('content_filter',1,False),(None,10,True)])
def test_provider_finish_reason_has_priority(reason,count,expected):
    result=generation_finish(reason,count,10)
    assert result['truncated'] is expected
    if reason is not None:assert result['finish']==reason and result['finish_source']=='provider'


@pytest.mark.parametrize('layout',['single','shards','bin'])
def test_model_revision_supports_checkpoint_layouts_and_weight_changes(tmp_path,layout):
    (tmp_path/'config.json').write_text('{}');(tmp_path/'tokenizer.json').write_text('{}')
    name='pytorch_model.bin' if layout=='bin' else 'model.safetensors' if layout=='single' else 'model-00001-of-00001.safetensors'
    if layout=='shards':(tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer':name}}))
    weight=tmp_path/name;weight.write_bytes(b'first');before=model_revision(tmp_path)
    weight.write_bytes(b'second weight contents')
    assert model_revision(tmp_path)!=before


def test_device_auto_supports_cpu_and_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    for role in ['retriever','scorer','train','generator']:
        monkeypatch.delenv('REBIND_'+role.upper()+'_DEVICE',raising=False)
        assert resolve_device({},role)=='cpu'
    with pytest.raises(ValueError,match='unavailable'):resolve_device({'devices':{'scorer':'cuda:3'}},'scorer')
