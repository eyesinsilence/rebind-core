"""Regression probes call the deployed frontend, without model or benchmark files."""
import copy
import json

import pytest

from rebind_mvp import final_frontend as frontend
from rebind_mvp.schema import CandidateRegistry, InferenceExample


def branch_graph():
    return frontend.validate_graph({
        'variables': [
            {'var_id': 'answer', 'is_answer': True, 'description': 'birthplace'},
            {'var_id': 'x', 'anchor': 'X', 'description': 'group'},
            {'var_id': 'person', 'description': 'member'},
        ],
        'slots': [
            {'slot_id': 's0', 'ordered_arguments': ['x', 'person'], 'relation_text': 'member'},
            {'slot_id': 's1', 'ordered_arguments': ['person', 'answer'], 'relation_text': 'birthplace'},
        ],
        'constraints': [],
    }, 'Where was the member of X born?')


class BranchGenerator:
    tokenizer = type('Tokenizer', (), {'encode': lambda self, text: list(text)})()

    def __init__(self):
        self.calls = []
        self.context = {}
        self.extractions = []
        self.round = 0
        self.fail_bob = False

    def pack(self, docs):
        return '\n'.join(d['text'] for d in docs), docs

    def bindings(self, registry=None):
        return {'assignments': [
            {'x': 'X', 'person': 'Alice', 'answer': 'UNKNOWN'},
            {'x': 'X', 'person': 'Bob', 'answer': 'B-town'},
        ]}

    def json(self, prompt, **kwargs):
        if prompt.startswith('Propose'):
            return {'values': ['Alice', 'Bob']}
        if prompt.startswith('Extract'):
            task = json.loads(prompt.split('Relation task: ')[1].split('\nDocuments:')[0])
            self.extractions.append((self.round, task))
            if task['known_inputs'] == {'person': 'Bob'} and 'Bob was born in B-town.' in prompt:
                if self.fail_bob:
                    raise ValueError('deliberate extraction failure')
                return {'candidates': [{'surface': 'B-town', 'doc_id': 'bob', 'quote': 'Bob was born in B-town.'}]}
            return {'candidates': []}
        if prompt.startswith('Select'):
            return self.bindings()
        return {'answer': 'B-town'}


class BranchRetriever:
    def __init__(self, generator, repeated=False):
        self.generator = generator
        self.calls = []
        self.repeated = repeated

    def search(self, query):
        self.calls.append(query)
        self.generator.round = len(self.calls) - 1
        if 'Bob' in query:
            text, did = 'Bob was born in B-town.', 'bob'
        else:
            text = 'No birthplace evidence.'
            did = 'same' if self.repeated else str(len(self.calls))
        return [{'doc_id': did, 'title': 'update', 'text': text, 'offsets': [0, len(text)]}]


def run_branches(monkeypatch, tmp_path, method='json_fix', repeated=False, fail_bob=False):
    graph = branch_graph()
    monkeypatch.setattr(frontend, 'compile_graph', lambda *args: (copy.deepcopy(graph), {}))
    gen = BranchGenerator()
    gen.fail_bob = fail_bob
    ret = BranchRetriever(gen, repeated)
    config = {
        'paths': {'data_root': str(tmp_path)},
        'retrieval': {'max_query_calls_including_initial': 12},
        'models': {'answer_prompt': 'Answer: ', 'structured_state_tokens': 2048},
    }
    neural = type('Neural', (), {'update': lambda self, *args: gen.bindings()})()
    result = frontend.run_repaired(InferenceExample('q', 'Where was the member of X born?'), ret, gen, config, method, neural)
    return result, gen, ret


@pytest.mark.parametrize('method', ['json_fix', 'bp_rebind_fix', 'rebind_fix'])
def test_second_branch_action_drives_extraction_and_candidate_inputs(monkeypatch, tmp_path, method):
    result, gen, ret = run_branches(monkeypatch, tmp_path, method)
    bob_round = next(t for t in result['trace'] if t['action'] and t['action']['inputs'] == {'person': 'Bob'})
    tasks = [task for round_index, task in gen.extractions if round_index == bob_round['round']]
    assert any(task['known_inputs'] == {'person': 'Bob'} for task in tasks)
    assert tasks[0]['known_inputs'] == {'person': 'Bob'}
    candidate = next(c for c in bob_round['candidates']['answer'] if c['surface'] == 'B-town')
    assert candidate['input_values'] == {'person': 'Bob'}
    accepted = next(a for a in bob_round['proposal']['accepted'] if a['candidate_id'] == candidate['candidate_id'])
    assert accepted['input_values'] == {'person': 'Bob'}
    assert len(ret.calls) == 4  # initial, upstream relation, Alice, Bob; no empty-result retries
    assert bob_round['retrieval_attempt']['status'] == 'completed'
    action_task = bob_round['extraction_tasks'][0]
    assert action_task['for_action'] and action_task['completed'] and action_task['produced_candidates']
    assert action_task['input_values'] == bob_round['action']['input_values']


def test_empty_extraction_is_completed_and_identical_evidence_is_not_retried(monkeypatch, tmp_path):
    result, gen, ret = run_branches(monkeypatch, tmp_path, repeated=True)
    assert len(ret.calls) == 4
    seen = set()
    for row in result['trace']:
        for task in row['extraction_tasks']:
            if not task['attempted']:
                continue
            assert task['key'] not in seen
            seen.add(task['key'])
            if not task['produced_candidates']:
                assert task['completed'] and task['status'] == 'completed'
    # New visible evidence permits extraction again without repeating a past query.
    upstream = [task for row in result['trace'] for task in row['extraction_tasks'] if task['slot_id'] == 's0' and task['attempted']]
    assert len(upstream) == 2
    assert upstream[0]['evidence_version'] != upstream[1]['evidence_version']


def test_extraction_failure_is_distinct_from_retrieval_and_does_not_loop(monkeypatch, tmp_path):
    result, gen, ret = run_branches(monkeypatch, tmp_path, fail_bob=True)
    row = result['trace'][-1]
    task = row['extraction_tasks'][0]
    assert row['retrieval_attempt']['status'] == 'completed'
    assert task['for_action'] and not task['completed'] and task['status'] == 'failed'
    assert not task['produced_candidates'] and len(ret.calls) == 4
    assert row['policy_errors']


def test_unrelated_quote_cannot_ground_output_elsewhere_in_document():
    registry = CandidateRegistry([{'var_id': 'answer'}])
    doc = {'doc_id': 'd', 'title': 'update', 'text': 'Alice studied at East University. Bob founded West Labs.'}
    with pytest.raises(ValueError):
        frontend.add_candidate(registry, 'answer', 'West Labs', 'retrieved', 'employer', {'person': 'Alice'}, 0,
                               doc, 'Alice studied at East University.')


def test_literal_cooccurrence_never_becomes_relation_verification():
    registry = CandidateRegistry([{'var_id': 'answer'}])
    doc = {'doc_id': 'd', 'title': 'update', 'text': 'Alice studied at East University. Bob founded West Labs.'}
    cid = frontend.add_candidate(registry, 'answer', 'West Labs', 'retrieved', 'employer', {'person': 'Alice'}, 0, doc, doc['text'])
    candidate = registry.pool['answer'][cid]
    assert candidate['verified'] is False
    assert candidate['literal_provenance']
    assert candidate['verification_level'] == 'literal_provenance'
    assert candidate['relation_checked'] is False and candidate['verified'] is False


def test_quote_offsets_are_document_absolute():
    registry = CandidateRegistry([{'var_id': 'answer'}])
    quote = 'Bob was born in B-town.'
    doc = {'doc_id': 'd', 'title': 'update', 'text': quote, 'offsets': [100, 100 + len(quote)]}
    cid = frontend.add_candidate(registry, 'answer', 'B-town', 'retrieved', 'birthplace', {'person': 'Bob'}, 0, doc, quote)
    source = registry.pool['answer'][cid]['provenance_records'][0]
    assert source['char_start'] == 100 and source['char_end'] == 100 + len(quote)
    assert source['quote'] == quote


def test_tasks_use_action_input_even_after_binding_changes_and_deduplicate():
    graph = branch_graph()
    action = frontend.next_action(graph, [{'x': 'UNKNOWN', 'person': 'Bob'}], set(), 'old')
    assignments = [{'x': 'X', 'person': 'Alice', 'answer': answer} for answer in ['A', 'B']]
    tasks = frontend.extraction_tasks(graph, assignments, action, 'new')
    assert [(t['slot_id'], t['input_values']) for t in tasks] == [
        ('s1', {'person': 'Bob'}), ('s0', {'x': 'X'}), ('s1', {'person': 'Alice'})]
    assert all(t['evidence_version'] == 'new' for t in tasks)
    assert tasks[0]['query_key'] == action['query_key'] and tasks[0]['key'] != action['key']
    assert frontend.next_action(graph, [{'person': 'Bob'}], {action['query_key']}, 'new') is None


def test_evidence_version_tracks_content_and_span_not_order():
    first = {'doc_id': 'a', 'title': 'A', 'text': 'Alice', 'offsets': [0, 5]}
    second = {'doc_id': 'b', 'title': 'B', 'text': 'Bob', 'offsets': [0, 3]}
    version = frontend.visible_evidence_version([first, second])
    assert version == frontend.visible_evidence_version([second, first])
    assert version != frontend.visible_evidence_version([first, dict(second, offsets=[10, 13])])
    assert version != frontend.visible_evidence_version([first, dict(second, text='Ben')])


def test_repeated_candidate_preserves_distinct_quote_records():
    registry = CandidateRegistry([{'var_id': 'answer'}])
    text = 'Bob was born in B-town. Bob has birthplace B-town.'
    doc = {'doc_id': 'd', 'title': 'update', 'text': text, 'offsets': [0, len(text)]}
    first_id = frontend.add_candidate(registry, 'answer', 'B-town', 'retrieved', 's1', {'person': 'Bob'}, 0,
                                      doc, 'Bob was born in B-town.')
    cid = frontend.add_candidate(registry, 'answer', 'B-town', 'retrieved', 's1', {'person': 'Bob'}, 1,
                                 doc, 'Bob has birthplace B-town.')
    assert cid == first_id
    assert len(registry.pool['answer'][cid]['provenance_records']) == 2
    assert len(registry.pool['answer'][cid]['origin_span_ids']) == 2


def test_execution_audit_checks_real_trace_and_rejects_mismatched_inputs(monkeypatch, tmp_path):
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / 'scripts/audit_frontend_execution.py'
    spec = importlib.util.spec_from_file_location('execution_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result, _, _ = run_branches(monkeypatch, tmp_path, repeated=True)
    assert module.audit([result])['status'] == 'passed'
    altered = copy.deepcopy(result)
    altered['trace'][-1]['extraction_tasks'][0]['input_values'] = {'person': 'Alice'}
    assert module.audit([altered])['status'] == 'failed'
    assert module.audit([{'trace': [{}]}])['status'] == 'no_auditable_traces'
