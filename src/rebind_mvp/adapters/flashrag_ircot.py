"""Execute the locked upstream IRCOT run_batch verbatim with local loading shims."""
import ast, pathlib
from types import SimpleNamespace

def native_class(path):
    source=pathlib.Path(path).read_text(); tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='IRCOTPipeline')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='run_batch')
    code=ast.Module(body=[method],type_ignores=[]); namespace={}; exec(compile(code,str(path),'exec'),namespace)
    return namespace['run_batch']

class NativeTemplate:
    def __init__(self,generator,root):
        self.generator=generator; self.exposures=[]
        tree=ast.parse((pathlib.Path(root)/'upstream/FlashRAG/flashrag/pipeline/active_pipeline.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='IRCOTPipeline')
        values={x.targets[0].id:ast.literal_eval(x.value) for x in cls.body if isinstance(x,ast.Assign)}
        self.instruction=values['IRCOT_INSTRUCTION']; self.example=values['IRCOT_EXAMPLE']
    def get_string(self,question,retrieval_result,previous_gen=''):
        raw,visible=self.generator.pack(retrieval_result); self.exposures.append(visible)
        reference=''.join('Wikipedia Title: '+d['title']+'\n'+d['text']+'\n\n' for d in self.generator.pack(retrieval_result)[1])
        return self.instruction+'\n\n'+self.example+reference+'Question: '+question+'\nThought: '+previous_gen

def native_run(example,retriever,generator,root):
    start=len(retriever.calls); item=SimpleNamespace(question=example.question,output={})
    item.update_output=lambda key,value:item.output.update({key:value})
    pipeline=SimpleNamespace(retriever=retriever,generator=generator,prompt_template=NativeTemplate(generator,root),max_iter=2)
    method=native_class(pathlib.Path(root)/'upstream/FlashRAG/flashrag/pipeline/active_pipeline.py'); method(pipeline,[item])
    parser_path=pathlib.Path(root)/'upstream/FlashRAG/flashrag/utils/pred_parse.py'
    parser=next(node for node in ast.parse(parser_path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='ircot_pred_parse')
    namespace={};exec(compile(ast.Module(body=[parser],type_ignores=[]),str(parser_path),'exec'),namespace)
    item.pred=item.output['pred'];namespace['ircot_pred_parse']([item])
    trace=[]
    for i,call in enumerate(retriever.calls[start:]):
        visible=pipeline.prompt_template.exposures[i] if i<len(pipeline.prompt_template.exposures) else []
        trace.append(dict(query=call['query'],retrieved_documents=call['documents'],retrieved_ids=call['doc_ids'],seen_ids=[d['doc_id'] for d in visible],prompt_visible_ids=[d['doc_id'] for d in visible],visible_spans=[dict(doc_id=d['doc_id'],offsets=d['offsets']) for d in visible]))
    item.output['retrieval_trace']=trace
    return item.output
