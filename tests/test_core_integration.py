import itertools, torch, pytest
from rebind_mvp.pair_update import TriangleUpdate,masked_softmax
from rebind_mvp.bp import belief_propagation
from rebind_mvp.source_reader import ReBindModule
from rebind_mvp.schema import *
from rebind_mvp.train import set_loss,transition_loss
from test_reference_ops import triangle_reference,permute_candidates

def fixture(dtype=torch.float64):
    torch.manual_seed(17); n,k,d=3,3,8
    return torch.randn(n,n,k,k,d,dtype=dtype),torch.ones(n,k,dtype=torch.bool),torch.ones(n,n,n,dtype=torch.bool)

def test_production_reference():
    z,v,a=fixture(); m=TriangleUpdate(8).double()
    torch.testing.assert_close(m.message(z,v,a,identity=True),triangle_reference(z,v,a),atol=1e-12,rtol=1e-12)

def test_gated_chunk_and_permutation():
    z,v,a=fixture(); v[0,2]=False; m=TriangleUpdate(8,1).double()
    first=m.message(z,v,a); m.row_chunk=3
    torch.testing.assert_close(first,m.message(z,v,a))
    order=torch.tensor([[0,2,1],[0,1,2],[0,2,1]])
    torch.testing.assert_close(permute_candidates(first,order),m.message(permute_candidates(z,order),v.gather(1,order),a))

def test_empty_gradient():
    z,v,a=fixture(); z.requires_grad_(); v[:,1:]=False; m=TriangleUpdate(8).double()
    out=m.message(z,v,a); assert out.abs().sum()==0
    out.sum().backward(); assert torch.isfinite(z.grad).all()

def test_bp_tree_exact():
    torch.manual_seed(2); n,k=4,3; u=torch.randn(n,k,dtype=torch.float64)
    p=torch.randn(n,n,k,k,dtype=torch.float64); p=(p+p.permute(1,0,3,2))/2
    edges=torch.zeros(n,n,dtype=torch.bool)
    for i,j in [(0,1),(1,2),(1,3)]: edges[i,j]=edges[j,i]=True
    assignments=list(itertools.product(range(k),repeat=n))
    scores=torch.stack([sum(u[i,a[i]] for i in range(n))+sum(p[i,j,a[i],a[j]] for i in range(n) for j in range(i+1,n) if edges[i,j]) for a in assignments])
    probs=scores.softmax(0); exact=torch.stack([torch.stack([sum(probs[t] for t,a in enumerate(assignments) if a[i]==c) for c in range(k)]) for i in range(n)])
    got,_=belief_propagation(u,p,torch.ones(n,k,dtype=torch.bool),edges,steps=4)
    torch.testing.assert_close(got.exp(),exact,atol=1e-12,rtol=1e-12)

def test_registry_reserve_restore_and_source_identity():
    r=CandidateRegistry([{'var_id':'v'}],limit=2)
    a=r.add('v','Alex','doc1:Alex',[],0); b=r.add('v','Alex','doc2:Alex',[],1)
    assert a!=b and b in r.pool['v']; r.select('v',[b]); assert r.active['v']==['UNKNOWN',b]
    r.select('v',[a]); assert r.remap('v',['UNKNOWN',a])=={0:0,1:1}
    e=EvidenceStore(); d=Document('d','d','t','raw old fact',(0,12)); s=e.add(d); assert s==e.add(d) and len(e.spans)==1

def neural_fixture():
    torch.manual_seed(41); n,k,h=3,3,16
    return dict(candidate=torch.randn(n,k,h),roles=torch.randn(n,h),tokens=torch.randn(2,7,h),token_mask=torch.ones(2,7,dtype=torch.bool),links=torch.ones(2,n,k,dtype=torch.bool),valid=torch.ones(n,k,dtype=torch.bool),allowed=torch.ones(n,n,n,dtype=torch.bool))

def test_source_pair_gradient_and_idempotence():
    f=neural_fixture(); model=ReBindModule(16,16,2).eval(); out=model(**f); repeated=model(**f)
    torch.testing.assert_close(out['unary'],repeated['unary'])
    (out['unary'].square().mean()+out['pair'].square().mean()+out['source'].square().mean()+out['interpretation'].square().mean()).backward()
    for name,p in model.named_parameters():
        if name.startswith('frontier'): continue
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.source_attention.in_proj_weight.grad.abs().sum()>0
    assert model.triangle.left.weight.grad.abs().sum()>0

def test_all_masked_no_update_signal():
    logits=torch.randn(3,3,requires_grad=True); v=torch.ones(3,3,dtype=torch.bool)
    loss=set_loss(logits,v,v,torch.zeros(3,dtype=torch.bool)); assert loss==0; loss.backward(); assert logits.grad.abs().sum()==0

def test_revision_before_detached():
    b=torch.randn(2,3,requires_grad=True); a=torch.randn(2,3,requires_grad=True)
    loss=transition_loss({'unary':b},{'unary':a},{'revision':[(0,1,2)]}); loss.backward()
    assert b.grad is None; assert a.grad[0,1]>0 and a.grad[0,2]<0

def test_final_read_budget_and_off_equivalence():
    from rebind_mvp.evaluate import run_loop
    class Tokenizer:
        def encode(self,s,**kw): return s.split()
    class Gen:
        def __init__(self): self.calls=[]; self.tokenizer=Tokenizer()
        def pack(self,docs): return '\n'.join(d['text'] for d in docs),docs
        def json(self,prompt):
            self.calls.append(dict(input_tokens=1,output_tokens=1,cache_hit=False))
            if 'independent fields' in prompt: return dict(answer='final',citations=['d6'])
            return dict(thought='step',next_query='q'+str(len(self.calls)),answer_ready=False)
    class Ret:
        def __init__(self): self.calls=[]
        def search(self,q):
            self.calls.append(q); i=len(self.calls)
            return [dict(doc_id=f'd{i}',parent_doc_id=f'd{i}',title=f't{i}',text=f'raw {i}',offsets=[0,5])]
    cfg={'retrieval':{'max_query_calls_including_initial':6},'models':{'structured_state_tokens':1024}}
    a=run_loop(InferenceExample('id','question'),Ret(),Gen(),cfg,'ircot_common')
    b=run_loop(InferenceExample('id','question'),Ret(),Gen(),cfg,'off')
    assert a['query_count']==6 and 'raw 6' in a['final_prompt'] and 'd6' in a['trace'][-1]['seen_ids']
    assert a['trace']==b['trace'] and a['final_prompt']==b['final_prompt'] and a['answer']==b['answer']
    retriever=Ret(); direct=run_loop(InferenceExample('id','question'),retriever,Gen(),cfg,'direct_reader',fixed=a['trace'])
    assert direct['query_count']==6 and 'raw 6' in direct['final_prompt'] and not retriever.calls

@pytest.mark.parametrize('assignment',['malformed',{'answer':{'surface':'Mira'}}])
def test_malformed_editable_state_preserves_final_reader(monkeypatch,assignment):
    import rebind_mvp.evaluate as evaluation
    graph=QuestionGraph([dict(var_id='answer')],[],[]);registry=CandidateRegistry(graph.variables)
    monkeypatch.setattr(evaluation,'propose',lambda *a:(graph,registry,dict(accepted=[],rejected=[],visible=[])))
    class Gen:
        calls=[]
        tokenizer=type('Tokenizer',(),{'encode':lambda self,s,**kw:s.split()})()
        def pack(self,docs): return 'raw source',docs
        def json(self,prompt):
            if 'independent fields' in prompt: return dict(answer='Mira',citations=['d'])
            return dict(assignments=[assignment])
        def generate(self,*args,**kwargs): return ['next query']
    class Ret:
        def search(self,q): return [dict(doc_id='d',parent_doc_id='d',title='title',text='Mira leads.',offsets=[0,11])]
    result=evaluation.run_loop(InferenceExample('id','Who leads?'),Ret(),Gen(),{'retrieval':{'max_query_calls_including_initial':1},'models':{'structured_state_tokens':1024}},'json_rebind')
    assert result['answer']=='Mira' and result['fallback']==1 and not result['state_included']

def test_sandbox_blocks_credentials_network_writes():
    import shutil
    if not shutil.which('bwrap'): pytest.skip('Linux bubblewrap is not installed')
    from rebind_mvp.adapters.pyrag import SandboxedExecutor
    code="""
import os,socket
checks=[]
try: open('/home/sandbox-test-user/.ssh/id_rsa').read(); checks.append(False)
except OSError: checks.append(True)
try: open('/usr/rebind_forbidden','w'); checks.append(False)
except OSError: checks.append(True)
try: socket.create_connection(('1.1.1.1',80),timeout=.1); checks.append(False)
except OSError: checks.append(True)
final_answer=str(all(checks))
"""
    assert SandboxedExecutor().execute(code,None,None,[])['final_answer']=='True'

def test_precision_and_no_cross_candidate_bridge():
    z,v,a=fixture(torch.float32); m=TriangleUpdate(8)
    a.zero_(); a[0,2,1]=True; z.zero_(); z[0,1,1,1]=2; z[1,2,2,1]=3
    assert m.message(z,v,a,identity=True)[0,2,1,1].abs().sum()==0
    z,v,a=fixture(torch.float32); reference=triangle_reference(z.double(),v,a)
    error=(m.message(z,v,a,identity=True).double()-reference).abs().max()
    assert error<1e-5
    bf=(m.message(z.bfloat16(),v,a,identity=True).double()-reference).abs().max()
    assert bf<.05

def test_cache_prefix_isolation():
    from rebind_mvp.audit import digest
    from dataclasses import asdict
    ex=InferenceExample('opaque','question')
    private={'answers':['CANARY_A'],'future_documents':['future A'],'gold_decomposition':['hidden A']}
    before=digest(asdict(ex)); private.update(answers=['CANARY_B'],future_documents=['future B'],gold_decomposition=['hidden B'])
    assert digest(asdict(ex))==before
    key=lambda **kw:digest(dict(qid='opaque',split='train',revision='abc',prompt='prompt',**kw))
    assert key(evidence='docA')!=key(evidence='docB')


def test_masked_set_loss_matches_manual():
    x=torch.tensor([[1.,2.,3.]],requires_grad=True); valid=torch.ones_like(x,dtype=torch.bool); allowed=torch.tensor([[True,False,True]])
    loss=set_loss(x,allowed,valid,torch.tensor([True]))
    torch.testing.assert_close(loss,-torch.log(x.softmax(-1)[0,[0,2]].sum()))

def test_not_applicable_controls_binding_and_keeps_raw():
    f=neural_fixture(); before=f['tokens'].clone(); model=ReBindModule(16,16,2).eval()
    output=model(**f); output['unary'].square().sum().backward()
    assert model.interpretation_special.weight.grad is not None
    assert model.interpretation_special.weight.grad.abs().sum()>0
    with torch.no_grad(): model.interpretation_special.bias.copy_(torch.tensor([-50.,50.]))
    suppressed=model(**f)
    assert not torch.allclose(suppressed['unary'],output['unary'])
    torch.testing.assert_close(f['tokens'],before)

def test_partial_dates_do_not_invent_conflicts():
    from rebind_mvp.joint_decoder import constraint_penalty
    graph=QuestionGraph([],[],[dict(type='before',arguments=['a','b'])])
    domain={v:[dict(candidate_id=v,identity=v,surface=s)] for v,s in [('a','2020'),('b','2020-06-01')]}
    assert constraint_penalty(graph,domain,(0,0))==0
    domain['a'][0]['surface']='2021'
    assert constraint_penalty(graph,domain,(0,0))==float('-inf')


def test_real_proposer_gold_canary_and_candidate_cache(tmp_path):
    from rebind_mvp.proposal import propose
    import json
    class Generator:
        config={'paths':{'data_root':str(tmp_path)}}
        def __init__(self): self.prompts=[]
        def pack(self,docs): return str(docs),docs
        def json(self,prompt):
            self.prompts.append(prompt)
            if 'Parse only the question' in prompt: return {'variables':[dict(var_id='a',description='person',type='person',is_answer=True)],'slots':[],'constraints':[]}
            return {'candidates':[dict(var_id='a',surface='Mira',doc_id='d',quote='Mira leads.',type='person',scope='unknown')]}
    ex=InferenceExample('canary','Who leads?'); docs=[dict(doc_id='d',parent_doc_id='d',title='Biography',text='Mira leads.',offsets=[0,11])]
    private={'answer':'POISON_A','future':'DO_NOT_READ_A','decomposition':['SECRET_A']}
    a=Generator(); ga,ra,pa=propose(a,ex,docs)
    private.update(answer='POISON_B',future='DO_NOT_READ_B',decomposition=['SECRET_B'])
    b=Generator(); gb,rb,pb=propose(b,ex,docs)
    assert a.prompts==b.prompts and ra.snapshot()==rb.snapshot() and pa==pb
    assert 'POISON' not in str(a.prompts) and 'SECRET' not in str(a.prompts)
    chunk=dict(docs[0],doc_id='d:0:11')
    _,resolved,evidence=propose(Generator(),ex,[chunk],graph=ga)
    assert resolved.snapshot()['a'][1]['surface']=='Mira'
    assert evidence['accepted'][0]['doc_id']=='d:0:11'
    assert evidence['accepted'][0]['id_resolution']=='unique_visible_parent_and_exact_quote'
    _,ambiguous,evidence=propose(Generator(),ex,[chunk,dict(chunk,doc_id='d:20:31',offsets=[20,31])],graph=ga)
    assert len(ambiguous.snapshot()['a'])==1 and not evidence['accepted']


def test_frozen_old_read_remaps_ids_and_blocks_new_candidate_path(tmp_path):
    from rebind_mvp.transitions import NeuralState
    from rebind_mvp.schema import CandidateRegistry,QuestionGraph
    graph=QuestionGraph([dict(var_id='a',description='person',type='person',is_answer=True)],[],[])
    registry=CandidateRegistry(graph.variables,limit=4); registry.add('a','Mira',['Mira'],['d'],0)
    config={'paths':{'data_root':str(tmp_path)},'rebind':{'hidden_dim':16,'layers':2,'joint_beam':4}}
    checkpoint=tmp_path/'model.pt'; torch.save({'model':ReBindModule(d=16,layers=2).state_dict()},checkpoint)
    state=NeuralState(None,config,checkpoint,'frozen_old_read',device='cpu')
    class Features:
        def build(self,q,g,r,docs,p,cache_context):
            k=len(r.snapshot()['a']); s=len(docs)
            return dict(candidate=torch.randn(1,k,768),roles=torch.randn(1,768),tokens=torch.randn(s,8,768),token_mask=torch.ones(s,8,dtype=torch.bool),links=torch.ones(s,1,k,dtype=torch.bool),valid=torch.ones(1,k,dtype=torch.bool),allowed=torch.ones(1,1,1,dtype=torch.bool),source_doc_index=torch.arange(s))
    state.features=Features(); ex=InferenceExample('id','Who leads?'); docs=[dict(doc_id='old')]
    first=state.update(ex,graph,registry,{},docs,0); old_domain=registry.snapshot()['a']; saved=state.old_reads['old'][0]['score'].clone()
    registry.add('a','Jonas',['Jonas'],['new'],1)
    second=state.update(ex,graph,registry,{},docs+[dict(doc_id='new')],1)
    values=second['source_scores']['old:window0'][0]; current=registry.snapshot()['a']
    for index,candidate in enumerate(current):
        original=next((i for i,x in enumerate(old_domain) if x['candidate_id']==candidate['candidate_id']),None)
        assert values[index]==(-1e4 if original is None else float(saved[0,original]))
    torch.testing.assert_close(state.old_reads['old'][0]['score'],saved)


def test_frontier_dependency_uses_bound_parameters_without_claiming_observed():
    from rebind_mvp.frontier import choose_frontier
    graph=QuestionGraph([], [dict(slot_id='s0',ordered_arguments=['film','director'],relation_text='director',order=0,dependencies=[]),dict(slot_id='s1',ordered_arguments=['director','birth'],relation_text='born',order=1,dependencies=['s0'])],[])
    state={'film':'Film X','director':'Mira','birth':'UNKNOWN'}
    slot,query=choose_frontier(graph,[state],None,['director Film X Mira'])
    assert slot=='s1' and query=='born Mira'


def test_explicit_identity_role_and_mutual_exclusion_constraints():
    from rebind_mvp.joint_decoder import constraint_penalty
    domain={'a':[dict(candidate_id='a1',identity='d1',surface='Mira',role='current',role_verified=True,entity_identity_verified=True,entity_id='person1',type='person')],'b':[dict(candidate_id='b1',identity='d2',surface='Jonas',entity_identity_verified=True,entity_id='person2',type='person')]}
    for constraint in [dict(type='equality',arguments=['a','b']),dict(type='required_role',arguments=['a'],role='founding'),dict(type='allowed_types',arguments=['a'],types=['date']),dict(type='forbidden_assignment',candidate_ids={'a':'a1','b':'b1'})]:
        assert constraint_penalty(QuestionGraph([],[],[constraint]),domain,(0,0))==float('-inf')
    domain['b'][0]['entity_identity_verified']=False
    assert constraint_penalty(QuestionGraph([],[],[dict(type='equality',arguments=['a','b'])]),domain,(0,0))==0


def test_real_source_reader_padding_invariance():
    from torch.nn import functional as F
    f=neural_fixture(); f['valid'][0,2]=False;f['links'][:,0,2]=False
    model=ReBindModule(16,16,2).eval(); original=model(**f)
    padded=dict(f,candidate=F.pad(f['candidate'],(0,0,0,2)),valid=F.pad(f['valid'],(0,2)),links=F.pad(f['links'],(0,2)))
    extra=model(**padded)
    torch.testing.assert_close(original['unary'],extra['unary'][:,:3],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(original['pair'],extra['pair'][:,:,:3,:3],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(original['source_special'],extra['source_special'],atol=1e-6,rtol=1e-5)


def test_fewshot_only_changes_state_prompt(monkeypatch):
    from rebind_mvp.fewshot import FewShotGenerator
    from rebind_mvp.proposal import Generator
    from rebind_mvp.evaluate import JSON_STATE_PROMPT
    calls=[]
    monkeypatch.setattr(Generator,'json',lambda self,prompt,**kwargs:calls.append((prompt,kwargs)) or {})
    generator=FewShotGenerator.__new__(FewShotGenerator);generator.demonstrations='TRAIN_ONLY_DEMONSTRATION'
    generator.json(JSON_STATE_PROMPT+'CURRENT_INPUT',schema={'type':'object'})
    generator.json('Question graph input');generator.json('Final reader raw documents')
    assert 'TRAIN_ONLY_DEMONSTRATION' in calls[0][0] and calls[0][0].endswith('CURRENT_INPUT')
    assert calls[0][1]['schema']=={'type':'object'}
    assert calls[1][0]=='Question graph input' and calls[2][0]=='Final reader raw documents'


@pytest.mark.parametrize('failure',[False,True])
def test_shared_proposals_and_failures_are_replayed(monkeypatch,failure):
    import rebind_mvp.evaluate as evaluation
    graph=QuestionGraph([dict(var_id='answer')],[],[]);registry=CandidateRegistry(graph.variables);proposals=[]
    docs=[dict(doc_id='d',parent_doc_id='d',title='title',text='Mira leads.',offsets=[0,11])]
    def propose(*args):
        proposals.append(1)
        if failure:raise ValueError('shared malformed proposal')
        return graph,registry,dict(accepted=[],rejected=[],visible=docs)
    monkeypatch.setattr(evaluation,'propose',propose)
    class Gen:
        calls=[]
        tokenizer=type('Tokenizer',(),{'encode':lambda self,s,**kw:s.split()})()
        def pack(self,docs):return 'raw source',docs
        def json(self,prompt):
            if 'independent fields' in prompt:return dict(answer='Mira',citations=['d'])
            return dict(assignments=[{'answer':'UNKNOWN'}])
        def generate(self,*args,**kwargs):return ['next query']
    class Ret:
        def search(self,q):return docs
    cfg={'retrieval':{'max_query_calls_including_initial':1},'models':{'structured_state_tokens':1024}};cache={}
    first=evaluation.run_loop(InferenceExample('id','Who leads?'),Ret(),Gen(),cfg,'json_rebind',proposal_cache=cache)
    registry.add('answer','Mira','identity',[],0)
    second=evaluation.run_loop(InferenceExample('id','Who leads?'),Ret(),Gen(),cfg,'json_rebind',proposal_cache=cache)
    assert len(proposals)==1
    assert first['trace'][0].get('candidates')==second['trace'][0].get('candidates')
    assert first['fallback']==second['fallback']==int(failure)
    assert first['final_prompt']==second['final_prompt']


def test_query_replay_calls_retriever_and_controls_termination():
    from rebind_mvp.evaluate import run_loop
    class Gen:
        def __init__(self): self.calls=[];self.tokenizer=type('T',(),{'encode':lambda self,s,**kw:s.split()})()
        def pack(self,docs):return str(docs),docs
        def json(self,prompt):
            if 'independent fields' in prompt:return dict(answer='done',citations=[])
            return dict(thought='ready early',answer_ready=True,next_query='ignored')
    class Ret:
        def __init__(self):self.queries=[]
        def search(self,q):
            self.queries.append(q);return [dict(doc_id=q,parent_doc_id=q,title=q,text=q,offsets=[0,len(q)])]
    cfg={'retrieval':{'max_query_calls_including_initial':6},'models':{'structured_state_tokens':1024}}
    ret=Ret();r=run_loop(InferenceExample('id','initial'),ret,Gen(),cfg,'ircot_common',query_schedule=['initial','replayed next','replayed last'])
    assert ret.queries==['initial','replayed next','replayed last']
    assert len(r['trace'])==3 and 'replayed last' in r['final_prompt']
    ret=Ret();r=run_loop(InferenceExample('id','initial'),ret,Gen(),cfg,'ircot_common',force_rounds=3)
    assert len(ret.queries)==3 and ret.queries[1]=='ignored'
    assert len(run_loop(InferenceExample('id','initial'),Ret(),Gen(),cfg,'ircot_common')['trace'])==1


def test_trace_candidates_are_snapshots(monkeypatch):
    import rebind_mvp.evaluate as evaluation
    graph=QuestionGraph([dict(var_id='answer')],[],[]);registry=CandidateRegistry(graph.variables)
    registry.add('answer','Mira','identity',['old'],0)
    def propose(*args):
        if args[-1]:registry.add('answer','Mira','identity',['new'],1)
        return graph,registry,dict(visible=[])
    monkeypatch.setattr(evaluation,'propose',propose)
    class Gen:
        calls=[];tokenizer=type('T',(),{'encode':lambda self,s,**kw:s.split()})()
        def pack(self,docs):return 'raw',docs
        def json(self,prompt):return dict(answer='Mira',citations=[],thought='',next_query='next',answer_ready=False)
    class Neural:
        def update(self,*args):return dict(assignments=[dict(answer='Mira')])
    class Ret:
        def search(self,q):return [dict(doc_id='d',title='d',text='Mira',offsets=[0,4])]
    r=evaluation.run_loop(InferenceExample('id','q'),Ret(),Gen(),{'retrieval':{'max_query_calls_including_initial':2},'models':{'structured_state_tokens':1024}},'rebind',Neural(),query_schedule=['q','next'])
    assert r['trace'][0]['candidates']['answer'][1]['origin_span_ids']==['old']
    assert set(r['trace'][1]['candidates']['answer'][1]['origin_span_ids'])=={'old','new'}


def test_common_slot_uses_only_own_arguments_and_verbalizer_preserves_them():
    from rebind_mvp.frontier_diagnostics import policy
    graph=QuestionGraph([], [dict(slot_id='s1',ordered_arguments=['person','school'],relation_text='graduated from')],[])
    state=dict(query_assignments=[dict(person='Alice',school='UNKNOWN')])
    class Gen:
        def __init__(self,reply):self.reply=reply
        def json(self,prompt,**kwargs):return self.reply
    common=dict(trace=[dict(query='initial'),dict(query='where did Bob graduate')])
    slot,query,audit=policy('common_slot',Gen(dict(slot_id='s1')),common)(InferenceExample('id','q'),graph,state,[], 'raw',0,'s1','graduated from Alice')
    assert query=='graduated from Alice' and 'Bob' not in query
    slot,query,audit=policy('verbalize',Gen(dict(next_query='where did Bob graduate')),common)(InferenceExample('id','q'),graph,state,[], 'raw',0,'s1','graduated from Alice')
    assert query=='graduated from Alice' and audit['fallback']
