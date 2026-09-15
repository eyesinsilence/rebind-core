import pytest
import torch
from rebind_mvp.adapt import BatchedReBind,loss_and_metrics
from rebind_mvp.source_reader import ReBindModule

@pytest.mark.parametrize('mode',['rebind','bp_rebind','independent_binding'])
def test_batched_forward_and_gradients_match(mode):
    torch.manual_seed(901);torch.set_num_threads(2)
    old=ReBindModule(input_dim=12,d=8,layers=2,mode=mode).double()
    new=BatchedReBind(input_dim=12,d=8,layers=2,mode=mode).double();new.load_state_dict(old.state_dict())
    b,n,k,s,t=2,3,3,2,4
    f=dict(candidate=torch.randn(b,n,k,12,dtype=torch.double),roles=torch.randn(b,n,12,dtype=torch.double),tokens=torch.randn(b,s,t,12,dtype=torch.double),token_mask=torch.ones(b,s,t,dtype=torch.bool),links=torch.ones(b,s,n,k,dtype=torch.bool),valid=torch.ones(b,n,k,dtype=torch.bool),allowed=torch.ones(b,n,n,n,dtype=torch.bool))
    f['valid'][:,1,-1]=False;f['links'][:,0,0,-1]=False;f['links'] &= f['valid'][:,None]
    # Fixed literal anchors exclude UNKNOWN even when legacy source links include it.
    f['valid'][:,1,0]=False;f['links'][:,:,1,0]=True
    actual=new(**f); expected=[old(**{key:value[i] for key,value in f.items()}) for i in range(b)]
    for key in ['unary','pair','interpretation','source','frontier']:
        torch.testing.assert_close(actual[key],torch.stack([r[key] for r in expected]),atol=1e-8,rtol=1e-7)
    loss=sum(actual[key][torch.isfinite(actual[key])].square().mean() for key in ['unary','pair','interpretation'])
    ref=sum(torch.stack([r[key] for r in expected])[torch.isfinite(actual[key])].square().mean() for key in ['unary','pair','interpretation'])
    loss.backward();ref.backward()
    for (name,p),(other,q) in zip(new.named_parameters(),old.named_parameters()):
        assert name==other
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)


def test_revision_supervision_changes_gradient():
    torch.manual_seed(7);b,n,k,s=2,3,5,3
    def output():return dict(unary=torch.randn(b,n,k,requires_grad=True),pair=torch.randn(b,n,n,k,k,requires_grad=True),interpretation=torch.randn(b,s,n,k+2,requires_grad=True))
    before=output();after=output();labels=dict(before=torch.zeros(b,n,dtype=torch.long),after=torch.ones(b,n,dtype=torch.long),before_interpretation=torch.zeros(b,s,n,dtype=torch.long),after_interpretation=torch.ones(b,s,n,dtype=torch.long))
    no,_=loss_and_metrics(before,after,labels,0);yes,_=loss_and_metrics(before,after,labels,.5)
    ga=torch.autograd.grad(no.sum(),after['unary'],retain_graph=True)[0];gb=torch.autograd.grad(yes.sum(),after['unary'])[0]
    assert not torch.allclose(ga,gb)
    assert torch.all(gb[...,1]<ga[...,1])


def test_adapted_open_qa_uses_public_holdout_only_and_resumes(tmp_path,monkeypatch):
    import json,pathlib,numpy as np
    import rebind_mvp.mquake as mq
    import rebind_mvp.proposal as proposal
    import rebind_mvp.retrieval as retrieval
    import rebind_mvp.transitions as transitions
    import rebind_mvp.evaluate as evaluation
    data=tmp_path/'data'
    for folder in ['adapt','public','memory']:(data/folder).mkdir(parents=True)
    (tmp_path/'configs').mkdir();(tmp_path/'runs').mkdir()
    (tmp_path/'configs/adapt_continue.yaml').write_text('adapt:\n  continue_pairs: []\n')
    (data/'adapt/split.json').write_text(json.dumps({'split':{'train':[2],'dev':[3],'test':[1]}}))
    public=[dict(qid=f'T_000001_q{i}',case_id=1,variant=i,question=f'question{i}',allowed_doc_ids=['update']) for i in range(3)]
    (data/'public/T.json').write_text(json.dumps(public));(data/'memory/T.json').write_text(json.dumps([dict(doc_id='update',text='only supplied update')]))
    np.save(data/'memory/T_e5.npy',np.zeros((1,768)))
    frozen=tmp_path/'frozen'
    for method in ['rebind','bp_rebind','independent_binding','no_revision_loss']:
        for seed in [17,29,43]:
            path=data/'adapt/checkpoints'/method/str(seed);path.mkdir(parents=True)
            (path/'best.pt').write_text('adapted');(path/'complete.json').write_text(json.dumps(dict(best_epoch=3,identity='training')))
            path=frozen/method/str(seed);path.mkdir(parents=True);(path/'best.pt').write_text('frozen')
    source=tmp_path/'src/rebind_mvp';source.mkdir(parents=True)
    for name in ['evaluate.py','proposal.py','transitions.py','source_reader.py','pair_update.py','bp.py','frontier.py','schema.py','retrieval.py']:(source/name).write_text('fixture')
    class Generator:
        def __init__(self,c):self.calls=[]
        def json(self,prompt):return {'answer':'test','citations':[]}
    class Encoder:
        def __init__(self,*a,**k):pass
    seen=[]
    def run_loop(ex,retriever,generator,c,method,neural):
        assert ex.qid.startswith('T_000001_')
        assert [d['text'] for d in retriever.docs]==['only supplied update']
        seen.append(ex.qid);return dict(answer='test',trace=[],eval_status='ok',fallback=0)
    monkeypatch.setattr(proposal,'Generator',Generator);monkeypatch.setattr(retrieval,'E5',Encoder)
    monkeypatch.setattr(transitions,'NeuralState',lambda *a,**k: object());monkeypatch.setattr(evaluation,'run_loop',run_loop)
    c=dict(paths=dict(workdir=str(tmp_path),data_root=str(data)),models=dict(retriever_path='unused'),retrieval=dict(top_k=1),mquake=dict(checkpoint_root=str(frozen)))
    result=mq.adapt_evaluate(c);assert result['actual']==result['expected']==45 and result['failures']==0
    assert len(seen)==42
    assert not (data/'private').exists()
    again=mq.adapt_evaluate(c);assert again==result and len(seen)==42
