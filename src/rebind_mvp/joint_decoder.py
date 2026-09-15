import torch

def decode(unary,pair,valid,beam=8,constraint=None):
    """Beam assignments; each unordered pair counted exactly once."""
    unary=unary.detach().cpu(); pair=pair.detach().cpu(); valid=valid.cpu()
    beams=[(0.,())]
    for i in range(len(unary)):
        expanded=[]
        for score,assignment in beams:
            for a in torch.where(valid[i])[0].tolist():
                trial=assignment+(a,)
                extra=0. if constraint is None else constraint(trial)
                if extra==float('-inf'): continue
                value=score+float(unary[i,a])+sum(float(pair[j,i,b,a]) for j,b in enumerate(assignment))+extra
                expanded.append((value,trial))
        beams=sorted(expanded,key=lambda x:(-x[0],x[1]))[:beam]
    return beams

def constraint_penalty(graph,domain,prefix):
    import datetime,re,calendar
    variables=list(domain); chosen={v:domain[v][prefix[i]] for i,v in enumerate(variables[:len(prefix)])}
    for v,candidate in chosen.items():
        for upstream,cid in candidate.get('input_candidate_ids',{}).items():
            if upstream in chosen and chosen[upstream]['candidate_id']!=cid:return float('-inf')
        for upstream,surface in candidate.get('input_values',{}).items():
            if upstream in chosen and chosen[upstream]['surface']!=surface:return float('-inf')
    for condition in graph.constraints:
        if not isinstance(condition,dict): continue
        kind=condition.get('type'); args=condition.get('arguments',[])
        if kind=='allowed_types' and len(args)==1 and args[0] in chosen:
            candidate=chosen[args[0]]
            if candidate.get('type','unknown') not in ['unknown','entity'] and candidate.get('type') not in condition.get('types',[]): return float('-inf')
        if kind=='required_role' and len(args)==1 and args[0] in chosen:
            candidate=chosen[args[0]]
            if candidate.get('role_verified') and candidate.get('role')!=condition.get('role'): return float('-inf')
        if kind=='forbidden_assignment':
            assignment=condition.get('candidate_ids',{})
            if assignment and all(v in chosen and chosen[v]['candidate_id']==cid for v,cid in assignment.items()): return float('-inf')
        if len(args)!=2 or any(a not in chosen for a in args): continue
        a,b=[chosen[v] for v in args]
        if a['candidate_id']=='UNKNOWN' or b['candidate_id']=='UNKNOWN': continue
        if kind=='inequality' and (a['identity']==b['identity'] or (a.get('entity_identity_verified') and b.get('entity_identity_verified') and a.get('entity_id')==b.get('entity_id'))): return float('-inf')
        if kind=='equality' and a.get('entity_identity_verified') and b.get('entity_identity_verified') and a.get('entity_id')!=b.get('entity_id'): return float('-inf')
        # Distinct source-local mentions do not prove different entities.
        if kind in ['before','after']:
            sa,sb=a['surface'],b['surface']
            if re.fullmatch(r'\d{4}(-\d\d(-\d\d)?)?',sa) and re.fullmatch(r'\d{4}(-\d\d(-\d\d)?)?',sb):
                def interval(value):
                    fields=[int(x) for x in value.split('-')]; year=fields[0]; first_month=fields[1] if len(fields)>1 else 1; last_month=fields[1] if len(fields)>1 else 12
                    first_day=fields[2] if len(fields)>2 else 1; last_day=fields[2] if len(fields)>2 else calendar.monthrange(year,last_month)[1]
                    return datetime.date(year,first_month,first_day),datetime.date(year,last_month,last_day)
                try: alo,ahi=interval(sa); blo,bhi=interval(sb)
                except ValueError: continue
                if (kind=='before' and alo>=bhi) or (kind=='after' and ahi<=blo): return float('-inf')
    return 0.
