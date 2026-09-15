import argparse, json, os, pathlib, sys, time, traceback
import yaml
from .audit import write,append,digest,Blocked

COMMANDS=['preflight','audit-sources','prepare-data','build-index','audit-data','audit-retrieval','diagnose','prepare-transitions','smoke','train','evaluate','intervene','report','mquake']
def main():
    parser=argparse.ArgumentParser(description='ReBind-RAG actual experiment stages; status and command provenance are persisted.')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in COMMANDS:
        p=sub.add_parser(name); p.add_argument('--config',required=True); p.add_argument('--resume',action='store_true')
        if name in ['diagnose','smoke','prepare-transitions']: p.add_argument('--limit',type=int)
        if name in ['diagnose','smoke','evaluate','audit-retrieval']: p.add_argument('--datasets',nargs='+',choices=['2wiki','musique_ans'])
        if name in ['diagnose','smoke','evaluate']:
            p.add_argument('--methods',nargs='+'); p.add_argument('--seed',type=int,default=17)
        if name=='train': p.add_argument('--seeds',nargs='+',type=int)
        if name=='evaluate': p.add_argument('--protocol',choices=['fixed','open'],required=True)
        if name=='mquake':
            p.add_argument('--phase',choices=['prepare','smoke','evaluate','report'],required=True);p.add_argument('--protocol',choices=['open','fixed'],default='open');p.add_argument('--methods',nargs='+')
    args=parser.parse_args(); config=yaml.safe_load(pathlib.Path(args.config).read_text()); root=pathlib.Path(config['paths'].get('workdir') or pathlib.Path(args.config).resolve().parents[1]).resolve(); root.mkdir(exist_ok=True)
    config['paths'].update(workdir=str(root),data_root=config['paths'].get('data_root') or str(root/'data'),hf_home=config['paths'].get('hf_home') or str(root/'data/hf'))
    os.environ['HF_HOME']=config['paths']['hf_home']; os.environ['TOKENIZERS_PARALLELISM']='false'
    config['models'].setdefault('server_url','http://127.0.0.1:8132/v1')
    config['models'].setdefault('retriever_path',str(root/'assets/e5')); config['models'].setdefault('generator_path',str(root/'assets/Qwen2.5-7B-Instruct'))
    code_hash=digest({str(p.relative_to(root)):digest(p) for p in (root/'src').rglob('*.py')})
    import tarfile
    snapshot=root/'runs/source_snapshots'/(code_hash+'.tar.gz'); snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():
        with tarfile.open(snapshot,'w:gz') as archive:
            for folder in ['src','scripts','configs']:
                for path in (root/folder).rglob('*'):
                    if path.is_file() and path.suffix in ['.py','.sh','.yaml']: archive.add(path,arcname=str(path.relative_to(root)))
    row=dict(code_sha256=code_hash,source_snapshot=str(snapshot),command=[sys.executable,'-m','rebind_mvp.cli']+sys.argv[1:],start=time.time(),config_sha256=digest(config),arguments=vars(args),runtime_env={k:os.environ.get(k) for k in ['REBIND_SERVER_URLS','REBIND_WORKERS','REBIND_TRAIN_DEVICE','OMP_NUM_THREADS']})
    stage=args.command+('_'+args.protocol if args.command=='evaluate' else '')
    if args.command=='mquake':stage+='_'+args.phase+'_'+args.protocol
    if args.command=='train' and args.seeds: stage+='_'+('_'.join(map(str,args.seeds)))
    state=root/'manifests'/f'{stage}.json'
    outputs_by_stage={
        'prepare-data':['manifests/splits.json','reports/data_audit.json'],
        'audit-sources':['reports/related_work_audit.md','reports/external_audit.json'],
        'audit-data':['reports/data_audit.json'], 'audit-retrieval':['reports/retrieval_audit.json'],
        'build-index':['data/indexes/'+ds+'/manifest.json' for ds in config['retrieval']['datasets']],
        'diagnose':['runs/natural_diagnose_open/summary.json'],
        'train':['reports/natural_training.json'], 'prepare-transitions':['data/natural_transitions/manifest.json'],
        'evaluate_open':['runs/natural_evaluate_open/summary.json'], 'evaluate_fixed':['runs/natural_evaluate_fixed/summary.json']}
    if args.command=='train': outputs_by_stage[stage]=['reports/natural_training_seed'+str(seed)+'.json' for seed in (args.seeds or config['train']['seeds'])]+[str(p.relative_to(root)) for p in pathlib.Path(config['paths']['data_root']).glob('checkpoints/*/*/*.pt') if int(p.parent.name) in (args.seeds or config['train']['seeds'])]
    if args.resume and state.exists():
        saved=json.loads(state.read_text())
        hashes=saved.get('output_hashes',{})
        if {k:v for k,v in saved.get('arguments',{}).items() if k!='resume'}=={k:v for k,v in vars(args).items() if k!='resume'} and saved.get('status')=='complete' and saved.get('config_sha256')==digest(config) and saved.get('validated_code_sha256',saved.get('code_sha256'))==code_hash and hashes and all((root/p).exists() and digest(root/p)==h for p,h in hashes.items()):
            append(root/'runs/initial_20260914/commands.jsonl',dict(row,end=time.time(),exit_code=0,cache_hit=True,status='reused_verified_stage'))
            print('Reused verified stage',stage); return
    try:
        if args.command=='preflight':
            import importlib.metadata,platform,subprocess,torch,psutil
            env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),memory=dict(psutil.virtual_memory()._asdict()),paths=config['paths'])
            write(root/'manifests/environment_runtime.json',env)
            (root/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
            for p in config['paths'].values():
                if isinstance(p,str): pathlib.Path(p).mkdir(parents=True,exist_ok=True)
            (root/'configs/resolved.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            result=env
        elif args.command=='prepare-data':
            from .data import prepare; result=prepare(config)
        elif args.command=='build-index':
            from .retrieval import build; result=build(config)
        elif args.command=='mquake':
            from .mquake import run;result=run(config,args)
        else:
            from . import stages
            result=stages.run(args.command,config,args)
        if args.command=='train': outputs_by_stage[stage]=['reports/natural_training_seed'+str(seed)+'.json' for seed in (args.seeds or config['train']['seeds'])]+[str(p.relative_to(root)) for p in pathlib.Path(config['paths']['data_root']).glob('checkpoints/*/*/*.pt') if int(p.parent.name) in (args.seeds or config['train']['seeds'])]
        status='partial' if (isinstance(result,dict) and (result.get('blocked') or result.get('natural_training_status')=='not_run' or result.get('status','').startswith('controlled_'))) else 'complete'
        write(state,dict(status=status,arguments=vars(args),config_sha256=digest(config),code_sha256=code_hash,output_hashes={p:digest(root/p) for p in outputs_by_stage.get(stage,[]) if (root/p).exists()},result=result)); row['exit_code']=0
    except Exception as e:
        row.update(exit_code=2 if isinstance(e,Blocked) else 1,error=str(e),traceback=traceback.format_exc()); write(state,dict(status='blocked' if isinstance(e,Blocked) else 'failed',**row)); traceback.print_exc()
    finally:
        row['end']=time.time(); append(root/'runs/initial_20260914/commands.jsonl',row); print(json.dumps(row,ensure_ascii=False),flush=True)
    if row['exit_code']: sys.exit(row['exit_code'])
if __name__=='__main__': main()
