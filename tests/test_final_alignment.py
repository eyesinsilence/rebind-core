import copy
import torch
from rebind_mvp.final_alignment import prefix_labels,relation_key,collate,supervised_loss


def example():
    return {'graph':{'variables':[{'var_id':'answer'},{'var_id':'x','anchor':'X'},{'var_id':'c'}],'slots':[{'slot_id':'s0','relation_text':'citizenship','ordered_arguments':['x','c']},{'slot_id':'s1','relation_text':'head of state','ordered_arguments':['c','answer']}]},'candidates':{'answer':[{'surface':'UNKNOWN'},{'surface':'A','origin_kind':'retrieved','slot_id':'s1','input_values':{'c':'C'}}],'x':[{'surface':'UNKNOWN'},{'surface':'X'}],'c':[{'surface':'UNKNOWN'},{'surface':'C','origin_kind':'parametric_hypothesis','slot_id':'s0','input_values':{'x':'X'}}]}}


def test_only_semantic_direction_and_available_values_receive_labels():
    t=example();triples=[['X','country of citizenship','C'],['C','head of state','A']];labels,_=prefix_labels(t,triples)
    assert labels['mask'].tolist()==[True,False,True]
    t['graph']['slots'][1]['relation_text']='current leader'
    labels,audit=prefix_labels(t,triples);assert not labels['mask'][0] and audit[1]['status']=='unaligned_relation_or_direction'
    t=example();t['candidates']['c'][1]['surface']='D';labels,_=prefix_labels(t,triples)
    assert not labels['mask'].any() and not labels['allowed'][:,0].any()
    t=example();t['candidates']['answer'][1]['input_values']={'c':'D'};labels,_=prefix_labels(t,triples);assert not labels['mask'][0]
    assert relation_key('head_of_government')=='head of government' and relation_key('leader') is None


def test_future_reference_entities_never_expand_feature_domain():
    t=example();before=copy.deepcopy(t);labels,_=prefix_labels(t,[['X','country of citizenship','FUTURE_SECRET'],['FUTURE_SECRET','head of state','A']])
    assert t==before and not labels['mask'].any()


def test_padding_sources_and_missing_labels_have_finite_training_gradients():
    from rebind_mvp.adapt import BatchedReBind
    from rebind_mvp.source_reader import ReBindModule
    torch.manual_seed(17);t=example();labels,_=prefix_labels(t,[['X','country of citizenship','C'],['C','head of state','A']]);payload=[];features={}
    for j,s in enumerate([1,3]):
        f=dict(candidate=torch.randn(3,2,12),roles=torch.randn(3,12),tokens=torch.randn(s,4,12),token_mask=torch.ones(s,4,dtype=torch.bool),links=torch.zeros(s,3,2,dtype=torch.bool),valid=labels['valid'],allowed=torch.ones(3,3,3,dtype=torch.bool));f['links'][:,:,0]=True
        lab=copy.deepcopy(labels);lab.update(interpretation_allowed=torch.zeros(s,3,4,dtype=torch.bool),interpretation_mask=torch.zeros(s,3,dtype=torch.bool));features[str(j)]=f;payload.append(dict(feature=str(j),labels=lab))
    f,y=collate(payload,features,'cpu');model=BatchedReBind(input_dim=12,d=8,layers=2);out=model(**f);loss=supervised_loss(out,f,y);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    reference=ReBindModule(input_dim=12,d=8,layers=2);reference.load_state_dict(model.state_dict())
    for j in range(2):
        single=reference(**features[str(j)]);torch.testing.assert_close(out['unary'][j],single['unary'],atol=1e-6,rtol=1e-5)
