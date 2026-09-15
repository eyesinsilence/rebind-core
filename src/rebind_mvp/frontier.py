def choose_frontier(graph,assignments,registry,queried,observed_slots=()):
    byid={s['slot_id']:s for s in graph.slots}
    for assignment in assignments:
        for slot in sorted(graph.slots,key=lambda s:(s.get('order',0),s['slot_id'])):
            if slot['slot_id'] in observed_slots: continue
            # A dependency is usable when its endpoints are bound; this does not mark it observed.
            if not all(x in observed_slots or (x in byid and all(assignment.get(v,'UNKNOWN')!='UNKNOWN' for v in byid[x]['ordered_arguments'])) for x in slot.get('dependencies',[])): continue
            args=[assignment.get(v,'UNKNOWN') for v in slot['ordered_arguments']]
            if not any(a!='UNKNOWN' for a in args): continue
            query=slot['relation_text']+' '+' '.join(a for a in args if a!='UNKNOWN')
            if query not in queried: return slot['slot_id'],query
    return None,None
