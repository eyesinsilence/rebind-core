from __future__ import annotations
from dataclasses import dataclass, field, asdict
from .audit import digest

@dataclass(frozen=True)
class Document:
    doc_id: str
    parent_doc_id: str
    title: str
    text: str
    offsets: tuple[int,int]
    sentence_offsets: tuple = ()

@dataclass(frozen=True)
class InferenceExample:
    qid: str
    question: str
    visible_documents: tuple[Document,...] = ()

@dataclass(frozen=True)
class EvaluatorRecord:
    qid: str
    answers: tuple
    supporting_facts: tuple = ()
    supporting_doc_ids: tuple = ()
    evidences: tuple = ()
    gold_decomposition: tuple = ()
    grouping_metadata: dict = field(default_factory=dict)

@dataclass
class QuestionGraph:
    variables: list
    slots: list
    constraints: list = field(default_factory=list)

@dataclass(frozen=True)
class InterpretationVariable:
    slot_id: str
    span_id: str
    alternatives: tuple
    unresolved: str = 'UNRESOLVED'
    not_applicable: str = 'NOT_APPLICABLE_TO_THIS_ROLE'

class EvidenceStore:
    def __init__(self): self.spans={}
    def add(self, doc, start=0, end=None):
        end=len(doc.text) if end is None else end
        if not 0<=start<=end<=len(doc.text): raise ValueError('Invalid source offset')
        value=dict(doc_id=doc.doc_id,parent_doc_id=doc.parent_doc_id,char_start=start+doc.offsets[0],char_end=end+doc.offsets[0],text=doc.text[start:end])
        value['text_hash']=digest(value['text']); sid=digest(value)
        self.spans.setdefault(sid,value)
        return sid

class CandidateRegistry:
    def __init__(self, variables, limit=6):
        self.limit=limit; self.pool={v['var_id']:{} for v in variables}; self.active={v:[] for v in self.pool}
        for v in self.pool:
            self.pool[v]['UNKNOWN']={'candidate_id':'UNKNOWN','variable_id':v,'surface':'UNKNOWN','identity':None,'origin_span_ids':[],'availability_time':0,'type':'unknown'}
    def add(self, variable_id, surface, identity, origins, time, type='entity'):
        # No identity conflation by surface; caller must ground identity in source-local event/mention.
        cid=digest([variable_id,identity])[:24]
        p=self.pool[variable_id]
        if cid not in p: p[cid]=dict(candidate_id=cid,variable_id=variable_id,surface=surface,identity=identity,origin_span_ids=list(origins),availability_time=time,type=type)
        else: p[cid]['origin_span_ids']=sorted(set(p[cid]['origin_span_ids'])|set(origins))
        self.select(variable_id,sorted((key for key in p if key!='UNKNOWN'),key=lambda key:(-p[key]['availability_time'],key)))
        return cid
    def select(self, v, priority=()):
        p=self.pool[v]; ids=list(dict.fromkeys([c for c in priority if c in p and c!='UNKNOWN']+[c for c in p if c!='UNKNOWN']))
        self.active[v]=['UNKNOWN']+ids[:self.limit-1]
        return self.active[v]
    def snapshot(self):
        import copy
        return copy.deepcopy({v:[self.pool[v][c] for c in self.select(v,self.active[v])] for v in self.pool})
    def remap(self, v, old_ids): return {i:self.active[v].index(c) for i,c in enumerate(old_ids) if c in self.active[v]}

@dataclass
class JointState:
    assignments: list
    source_scores: dict
    status: dict
    revision_log: list
    candidate_ids: dict
