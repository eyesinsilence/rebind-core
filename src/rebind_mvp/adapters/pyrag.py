"""Official PyRAG controller, with a networkless read-only subprocess executor."""
import json,os,pathlib,selectors,subprocess,sys,time

SANDBOX=r'''
import json,sys,resource
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,)*2)
resource.setrlimit(resource.RLIMIT_CPU,(15,15))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
resource.setrlimit(resource.RLIMIT_NPROC,(8,8))
def rpc(kind,*args,**kwargs):
 print(json.dumps({'rpc':kind,'args':args,'kwargs':kwargs}),flush=True)
 reply=json.loads(sys.stdin.readline())
 if 'error' in reply: raise RuntimeError(reply['error'])
 return reply['result']
ns={'retrieve':lambda *a,**kw:rpc('retrieve',*a,**kw),'answer':lambda *a,**kw:rpc('answer',*a,**kw)}
try:
 exec(sys.argv[1],ns)
 print(json.dumps({'done':True,'final_answer':str(ns.get('final_answer','UNKNOWN')),'variables':{k:repr(v) for k,v in ns.items() if not k.startswith('__') and k not in ['retrieve','answer']}}),flush=True)
except Exception as e:
 print(json.dumps({'error':str(e)}),flush=True)
'''
class SandboxedExecutor:
    def execute(self,code,retrieve_fn,answer_fn,execution_log):
        cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--setenv','PATH','/usr/bin','--chdir','/tmp','/usr/bin/python3','-I','-c',SANDBOX,code]
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env={'PATH':'/usr/bin:/bin'})
        selector=selectors.DefaultSelector(); selector.register(p.stdout,selectors.EVENT_READ); started=time.time()
        try:
            while time.time()-started<300:
                events=selector.select(1)
                if not events:
                    if p.poll() is not None: break
                    continue
                line=p.stdout.readline()
                if not line: break
                try: message=json.loads(line)
                except json.JSONDecodeError: continue
                if message.get('done'): return dict(final_answer=message['final_answer'],variables=message['variables'],execution_log=execution_log)
                if 'error' in message: raise RuntimeError(message['error'])
                if message.get('rpc') in ['retrieve','answer']:
                    try:
                        fn=retrieve_fn if message['rpc']=='retrieve' else answer_fn
                        reply={'result':fn(*message['args'],**message['kwargs'])}
                    except Exception as e: reply={'error':str(e)}
                    p.stdin.write(json.dumps(reply)+'\n'); p.stdin.flush()
            raise RuntimeError('Sandbox failed or timed out: '+p.stderr.read(2000) if p.poll() is not None else 'Sandbox timed out')
        finally:
            if p.poll() is None: p.kill()
            p.wait(); selector.close()

def run(example,retriever,generator,root):
    sys.path.insert(0,str(pathlib.Path(root)/'upstream/PyRAG'))
    from pyrag.runner import RAGProgramRunner
    class LLM:
        def generate(self,system_prompt,user_prompt): return generator.generate([user_prompt],system=system_prompt)[0]
    class Retrieval:
        def __init__(self): self.count=0; self.trace=[]
        def retrieve(self,query,topk=5):
            if self.count>=6: raise RuntimeError('Shared six-query budget exhausted')
            self.count+=1
            docs=retriever.search(query); _,visible=generator.pack(docs)
            self.trace.append(dict(query=query,retrieved_documents=docs,retrieved_ids=[d['doc_id'] for d in docs],provided_ids=[d['doc_id'] for d in visible],visible_documents=visible,requested_topk=topk,actual_topk=len(docs)))
            return [d['title']+'\n'+d['text'] for d in visible]
    retrieval=Retrieval(); runner=RAGProgramRunner(LLM(),retrieval); runner.executor=SandboxedExecutor(); first=len(generator.calls)
    try: result=runner.run(example.question,topk=5)
    except RuntimeError as e: result=dict(final_answer='',eval_status='runtime_failure',error=str(e))
    prompts=[call['prompt'] for call in generator.calls[first:]]
    for row in retrieval.trace:
        visible=[d for d in row.pop('visible_documents') if any(d['text'] in prompt for prompt in prompts)]
        row.update(seen_ids=[d['doc_id'] for d in visible],prompt_visible_ids=[d['doc_id'] for d in visible],visible_spans=[dict(doc_id=d['doc_id'],offsets=d['offsets']) for d in visible])
    result['retrieval_trace']=retrieval.trace
    return result
