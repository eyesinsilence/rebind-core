import pytest
import pathlib,types
import numpy as np
from rebind_mvp.mquake import official_function,EditRetriever

ROOT=pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/'upstream/mquake_remastered').exists(), reason='Optional upstream MQuAKE code is not bundled')
def test_official_exact_alias_metric():
    check=official_function(ROOT,'check_answer')
    r=dict(answer='Old',answer_alias=[],new_answer='New York City',new_answer_alias=['NYC'])
    assert check(True,r,'nyc') and not check(True,r,'NYC.') and not check(True,r,None)
    assert check(False,r,'old') and not check(True,r,'old')


@pytest.mark.skipif(not (ROOT/'upstream/mquake_remastered').exists(), reason='Optional upstream MQuAKE code is not bundled')
def test_official_mask_excludes_conflicting_updates_without_adding_gold_path():
    mask=official_function(ROOT,'get_edits_without_contamination')
    rewrite=lambda target:dict(subject='City',relation_id='P6',prompt='{} is led by',target_new_str=target)
    a=dict(case_id=1,requested_rewrite=[rewrite('Ada')],new_triples=[['Q1','P6','QA']],new_triples_labeled=[['City','led by','Ada']])
    b=dict(case_id=2,requested_rewrite=[rewrite('Ben')])
    model=types.SimpleNamespace(dataset=[a,b],rand_list=[1,2])
    facts,*_=mask(model,a)
    assert facts==['City is led by Ada']
    a['answer']='POISON_ANSWER';a['new_answer']='POISON_NEW_ANSWER'
    assert mask(model,a)[0]==facts


def test_retrieval_cannot_reintroduce_masked_edits():
    encoder=types.SimpleNamespace(encode=lambda *a:np.array([[1.,0.]],dtype=np.float32))
    retriever=EditRetriever([{'doc_id':'allowed'},{'doc_id':'masked'}],np.array([[.1,0.],[1.,0.]]),['allowed'],encoder,5)
    assert [r['doc_id'] for r in retriever.search('public question')]==['allowed']
