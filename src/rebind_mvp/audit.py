from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, time
from dataclasses import asdict, is_dataclass

def digest(x):
    if isinstance(x, pathlib.Path):
        h=hashlib.sha256()
        with x.open('rb') as f:
            for b in iter(lambda:f.read(8<<20), b''): h.update(b)
        return h.hexdigest()
    return hashlib.sha256(json.dumps(asdict(x) if is_dataclass(x) else x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

def write(path, value):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n'); tmp.replace(path)

def append(path, value):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f: f.write(json.dumps(value,ensure_ascii=False,default=str)+'\n')

def execute(cmd, root, name, timeout=None):
    root=pathlib.Path(root); log=root/'runs/initial_20260914'/f'{name}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    start=time.time()
    with log.open('w') as f:
        p=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    row=dict(command=cmd,start=start,end=time.time(),exit_code=p.returncode,log=str(log),log_sha256=digest(log))
    append(root/'runs/initial_20260914/commands.jsonl',row)
    return row

class Blocked(RuntimeError): pass
