import copy,threading
import json,pathlib,time
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from .audit import digest,write,append
from .schema import Document
from .runtime import resolve_device

class E5:
    @property
    def tokenizer(self):
        if not hasattr(self.local,"tokenizer"): self.local.tokenizer=copy.deepcopy(self.tokenizer_base)
        return self.local.tokenizer
    def __init__(self,path,device=None,batch=64):
        device=resolve_device(role='retriever',override=device)
        if batch < 1: raise ValueError('Encoder batch size must be positive')
        self.local=threading.local(); self.tokenizer_base=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).eval().to(device)
        self.model.requires_grad_(False); self.device=device; self.batch=batch
        self.embedding_dim=int(self.model.config.hidden_size)
    @staticmethod
    def prefix(text,kind):
        while text.startswith(('query: ','passage: ')): text=text.split(': ',1)[1]
        return kind+': '+text
    @torch.inference_mode()
    def encode(self,texts,kind='passage',tokens=False):
        vectors=[]; token_batches=[]; mask_batches=[]
        for i in range(0,len(texts),self.batch):
            batch=[self.prefix(t,kind) for t in texts[i:i+self.batch]]
            inputs=self.tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
            out=self.model(**inputs).last_hidden_state; mask=inputs['attention_mask'].bool()
            if tokens:
                token_batches.append(out.detach());mask_batches.append(mask)
                continue
            mean=(out*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
            vectors.append(torch.nn.functional.normalize(mean,dim=-1).float().cpu().numpy())
        if tokens:
            if not token_batches:
                return torch.empty((0,0,self.embedding_dim),device=self.device),torch.empty((0,0),dtype=torch.bool,device=self.device)
            width=max(t.shape[1] for t in token_batches)
            padded=[torch.nn.functional.pad(t,(0,0,0,width-t.shape[1])) for t in token_batches]
            masks=[torch.nn.functional.pad(m,(0,width-m.shape[1]),value=False) for m in mask_batches]
            return torch.cat(padded),torch.cat(masks)
        return np.concatenate(vectors) if vectors else np.empty((0,self.embedding_dim),dtype=np.float32)

def build(config):
    root=pathlib.Path(config['paths']['workdir']); data=pathlib.Path(config['paths']['data_root']); encoder=E5(config['models']['retriever_path'],device=resolve_device(config,'retriever'))
    summaries={}
    for ds in config['retrieval']['datasets']:
        out=data/'indexes'/ds; out.mkdir(parents=True,exist_ok=True)
        chunks=[]; parents=0
        with (data/'corpus'/f'{ds}.jsonl').open() as f:
            for line in f:
                doc=json.loads(line); parents+=1; title=doc['title']; text=doc['text']
                title_tokens=len(encoder.tokenizer.encode('passage: '+title+'\n',add_special_tokens=False)); budget=512-title_tokens-2
                if budget<16: raise ValueError('Title exceeds E5 budget')
                offsets=encoder.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                i=0
                while i<max(1,len(offsets)):
                    stop=min(i+budget,len(offsets)); start=offsets[i][0] if offsets else 0; end=offsets[stop-1][1] if offsets else 0
                    while len(encoder.tokenizer.encode('passage: '+title+'\n'+text[start:end]))>512:
                        stop-=1; end=offsets[stop-1][1]
                    chunk=dict(doc); chunk.update(doc_id=doc['doc_id']+f':{start}:{end}',text=text[start:end],offsets=[start,end]); chunks.append(chunk)
                    i=max(i+1,stop)
        with (out/'chunks.jsonl').open('w') as f:
            for c in chunks: f.write(json.dumps(c,ensure_ascii=False)+'\n')
        vectors=np.lib.format.open_memmap(out/'vectors.npy',mode='w+',dtype='float32',shape=(len(chunks),encoder.embedding_dim))
        start=time.time()
        for i in range(0,len(chunks),256):
            vectors[i:i+256]=encoder.encode([c['title']+'\n'+c['text'] for c in chunks[i:i+256]])
            if i%4096==0: print(ds,i,len(chunks),'elapsed',round(time.time()-start),flush=True)
        vectors.flush()
        summaries[ds]=dict(parent_count=parents,chunk_count=len(chunks),manifest_sha256=digest(out/'chunks.jsonl'),vectors_sha256=digest(out/'vectors.npy'),seconds=time.time()-start,index='exact_numpy_inner_product',tie_break='stable_chunk_manifest_order')
        write(out/'manifest.json',summaries[ds])
    write(root/'reports/retrieval_audit.json',dict(build=summaries,status='built_validation_pending'))
    return summaries

class Retriever:
    def __init__(self,config,ds,encoder=None):
        self.encoder=encoder or E5(config['models']['retriever_path'],device=resolve_device(config,'retriever')); directory=pathlib.Path(config['paths']['data_root'])/'indexes'/ds
        self.docs=[json.loads(x) for x in (directory/'chunks.jsonl').read_text().splitlines()]
        self.vectors=np.load(directory/'vectors.npy',mmap_mode='r'); self.top_k=config['retrieval']['top_k']; self.calls=[]
        if self.vectors.shape != (len(self.docs),self.encoder.embedding_dim):
            raise ValueError('Index shape does not match corpus size and encoder dimension; rebuild this index')
    def search(self,q):
        t=time.time(); vector=self.encoder.encode([q],'query')[0]; scores=self.vectors@vector
        # stable sorting makes equal scores deterministic and matches brute force.
        ids=np.argsort(-scores,kind='stable')[:self.top_k]
        docs=[dict(self.docs[i],score=float(scores[i])) for i in ids]
        self.calls.append(dict(query=q,doc_ids=[d['doc_id'] for d in docs],documents=docs,seconds=time.time()-t))
        return docs
    def batch_search(self,questions,return_score=False):
        results=[self.search(q) for q in questions]; docs=[[dict(d,id=d['doc_id'],contents=d['title']+'\n'+d['text']) for d in row] for row in results]
        return (docs,[[d['score'] for d in row] for row in results]) if return_score else docs
