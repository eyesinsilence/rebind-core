"""Audit public frontend traces without reading answer labels or loading models."""
import argparse
import collections
import json
import pathlib


def audit(records):
    counts = collections.Counter()
    violations = []
    for result in records:
        counts['records'] += 1
        attempted = set()
        for row in result.get('trace', []):
            counts['rounds'] += 1
            if row.get('frontend_version') not in ['action_bound_literal_provenance_v2','candidate_identity_relation_constraints_v3']:
                counts['legacy_or_unsupported_rounds'] += 1
                continue
            counts['auditable_rounds'] += 1
            tasks = row.get('extraction_tasks', [])
            action = row.get('action')
            if action:
                counts['scheduled_actions'] += 1
                matching = [t for t in tasks if t.get('for_action') and t.get('query_key') == action.get('query_key')
                            and t.get('input_values') == action.get('input_values')]
                if len(matching) != 1 or not tasks[0].get('for_action') or matching[0].get('input_candidate_ids') != action.get('input_candidate_ids'):
                    violations.append('Scheduled action has no unique first extraction task with matching inputs')
                else:
                    counts['actions_with_matching_extraction'] += 1
            for task in tasks:
                if task.get('evidence_version') != row.get('evidence_version'):
                    violations.append('Extraction evidence version differs from visible evidence')
                if not task.get('attempted'):
                    counts['reused_extractions'] += 1
                    continue
                counts['extraction_attempts'] += 1
                if task['key'] in attempted:
                    violations.append('Same extraction task retried with unchanged evidence')
                attempted.add(task['key'])
                if not task.get('completed'):
                    counts['failed_extractions'] += 1
                elif task.get('produced_candidates'):
                    counts['extractions_with_candidates'] += 1
                else:
                    counts['completed_without_candidates'] += 1
            by_key = {t['key']: t for t in tasks}
            for accepted in row.get('proposal', {}).get('accepted', []):
                task = by_key.get(accepted.get('task_key'))
                if not task or task.get('input_values') != accepted.get('input_values'):
                    violations.append('Accepted candidate inputs differ from extraction inputs')
            for candidates in row.get('candidates', {}).values():
                for candidate in candidates:
                    if candidate.get('origin_kind') != 'retrieved':
                        continue
                    counts['retrieved_candidate_observations'] += 1
                    if candidate.get('verification_level') == 'literal_provenance':
                        counts['literal_only_observations'] += 1
                        if candidate.get('verified') or candidate.get('relation_checked'):
                            violations.append('Literal provenance was promoted to semantic verification')
    return dict(counts=dict(counts), violations=violations,
                status='failed' if violations else 'no_auditable_traces' if not counts['auditable_rounds'] else 'passed',
                scope='Execution consistency only; no gold-label coverage, revision correctness or QA estimates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=pathlib.Path, nargs='+', help='One full inference trace JSON per file')
    args = parser.parse_args()
    result = audit(json.loads(p.read_text(encoding='utf-8')) for p in args.traces)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
