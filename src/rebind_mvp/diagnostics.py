from .runtime import resolve_device,encoder_dimension,peak_memory
"""Program-verified numerical learning fixtures. Never reported as natural QA."""
import csv,json,pathlib,random,time
import numpy as np
import torch
from .audit import digest,write,append,Blocked
from .schema import QuestionGraph,CandidateRegistry
from .transitions import FeatureBuilder
from .source_reader import ReBindModule
from .train import set_loss,transition_loss

def prepare_transitions(c):
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); target=data/'transitions'; target.mkdir(exist_ok=True)
    from .retrieval import E5
    encoder=E5(c['models']['retriever_path'],device=resolve_device(c,'retriever')); builder=FeatureBuilder(encoder,c)
    examples=[]; rng=random.Random(719)
    names=['Mira','Jonas','Elena','Tariq','Nadia','Felix','Leila','Owen','Clara','Hugo','Sofia','Noah','Iris','Damon','Lena','Arun','Yara','Emil','Rina','Tomas','Anya','Malik','Esme','Ivan','Ada','Basil','Cora','Dario','Etta','Farid','Gia','Hadi']
    graph=QuestionGraph([dict(var_id='founder',description='person who led the project at its founding',type='person',is_answer=True),dict(var_id='current',description='current project lead',type='person'),dict(var_id='researcher',description='current researcher',type='person')],[dict(slot_id='founding',ordered_arguments=['founder'],relation_text='led at founding',dependencies=[],order=0)])
    for i in range(16):
        a,b=names[2*i:2*i+2]; org=f'Project {chr(65+i)}'; q=f'Who led {org} when it was founded?'
        founder=rng.choice([a,b]); current=a; researcher=b
        old=f'{org} biography: {a} is the project lead. {b} is a researcher on the project. This biography gives no date for these roles.'
        if founder==b: new=f'The biography describes current roles, not roles at founding. At the founding of {org}, {b} was the lead. {a} joined later and became the current lead.'
        else: new=f'The biography describes current roles. {a} has led {org} continuously since its founding. {b} joined later as a researcher.'
        before=[dict(doc_id=f'w{i}:old',parent_doc_id=f'w{i}:old',title=org,text=old,offsets=[0,len(old)],sentence_offsets=[])]
        after=before+[dict(doc_id=f'w{i}:scope',parent_doc_id=f'w{i}:scope',title=org+' role history',text=new,offsets=[0,len(new)],sentence_offsets=[])]
        registry=CandidateRegistry(graph.variables,limit=3)
        for v in registry.pool:
            for name in [a,b]: registry.add(v,name,[org,name],[f'w{i}:old'],0)
        f_before=builder.build(q,graph,registry,before,{})
        f_after=builder.build(q,graph,registry,after,{})
        index=next(j for j,entry in enumerate(registry.snapshot()['founder']) if entry['surface']==founder)
        examples.append(dict(candidate_surfaces={v:[x['surface'] for x in domain] for v,domain in registry.snapshot().items()},expected_founder=founder,group_id=f'world-{i}',split='micro_train' if i<12 else 'diagnostic_dev',question=q,before=f_before,after=f_after,target=index,target_indices=[next(j for j,entry in enumerate(registry.snapshot()[v]) if entry['surface']==name) for v,name in [('founder',founder),('current',a),('researcher',b)]],change_type='disambiguation' if founder!=a else 'confirmation',raw_before=before,raw_after=after,review_status='program_verified',candidate_policy='question-template domains + explicit raw names; separate controlled diagnostic'))
        print('prepared controlled world',i,flush=True)
    torch.save(examples,target/'controlled_micro.pt')
    audit=dict(controlled_worlds=16,train_worlds=12,dev_worlds=4,training_natural_transitions=0,revision_labels=0,disambiguation=sum(e['change_type']=='disambiguation' for e in examples),confirmation=sum(e['change_type']=='confirmation' for e in examples),weak_labels=0,human_reviewed=0,status='controlled_pipeline_only_natural_transitions_pending',limitations=['Before prefix permits either founder; therefore these are disambiguation/confirmation, not verified correction of a uniquely wrong prior binding.','Two text templates share structure; this dev split tests held-out names/worlds, not template-family generalization.'])
    write(root/'reports/label_audit.json',audit); return audit

def train_all(c):
    root=pathlib.Path(c['paths']['workdir']); data=pathlib.Path(c['paths']['data_root']); file=data/'transitions/controlled_micro.pt'
    if not file.exists(): raise Blocked('No prepared transitions')
    examples=torch.load(file,weights_only=True); train=examples[:12]; dev=examples[12:]; device=resolve_device(c,'train'); summaries=[]; curves=[]
    # These weights remain under diagnostics, never promoted to natural-data checkpoints.
    for method in ['rebind','bp_rebind','independent_binding','no_revision_loss']:
        torch.manual_seed(17); mode='rebind' if method=='no_revision_loss' else method
        model=ReBindModule(input_dim=examples[0]['after']['candidate'].shape[-1],d=c['rebind']['hidden_dim'],layers=c['rebind']['layers'],mode=mode).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=c['train']['learning_rate'],weight_decay=c['train']['weight_decay']); output=data/'diagnostic_checkpoints_v4'/method/'17'; output.mkdir(parents=True,exist_ok=True)
        best=float('inf'); start=time.time()
        for epoch in range(21):
            model.train(); losses=[]
            for example in train:
                opt.zero_grad(set_to_none=True); features={k:v.to(device) for k,v in example['after'].items()}; out=model(**features)
                valid=features['valid']; allowed=torch.zeros_like(valid); target_indices=example['target_indices']
                for vi,ci in enumerate(target_indices): allowed[vi,ci]=True
                mask=torch.ones(3,dtype=torch.bool,device=device)
                pairvalid=valid[:,None,:,None]&valid[None,:,None,:]; pairallowed=torch.zeros_like(pairvalid)
                for vi,ci in enumerate(target_indices):
                    for vj,cj in enumerate(target_indices): pairallowed[vi,vj,ci,cj]=True
                pairmask=torch.triu(torch.ones(3,3,dtype=torch.bool,device=device),diagonal=1)
                loss=set_loss(out['unary'],allowed,valid,mask)+set_loss(out['pair'].flatten(-2),pairallowed.flatten(-2),pairvalid.flatten(-2),pairmask)
                special=out['interpretation']; source_allowed=torch.zeros_like(special,dtype=torch.bool); source_valid=torch.ones_like(source_allowed)
                for si,targets in enumerate([[valid.shape[-1]+1,target_indices[1],target_indices[2]],[target_indices[0],target_indices[1],valid.shape[-1]+1]]):
                    for vi,ci in enumerate(targets): source_allowed[si,vi,ci]=True
                loss=loss+set_loss(special,source_allowed,source_valid,torch.ones(special.shape[:-1],dtype=torch.bool,device=device))
                if epoch:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                losses.append(float(loss.detach()))
            model.eval(); devloss=[]; correct=0; traincorrect=0
            with torch.no_grad():
                for idx,example in enumerate(train+dev):
                    features={k:v.to(device) for k,v in example['after'].items()}; out=model(**features); pred=int(out['unary'][0].argmax())
                    if idx<12: traincorrect+=pred==example['target']
                    else:
                        correct+=pred==example['target']; devloss.append(float(-out['unary'][0].log_softmax(-1)[example['target']]))
            curve=dict(method=method,seed=17,epoch=epoch,train_loss=float(np.mean(losses)),train_accuracy=traincorrect/12,dev_loss=float(np.mean(devloss)),dev_accuracy=correct/4,seconds=time.time()-start); curves.append(curve)
            if epoch%5==0: print(curve,flush=True)
            ckpt=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,config_hash=digest(c),data_hash=digest(file),method=method,seed=17,scope='controlled_micro_only')
            torch.save(ckpt,output/'last.pt')
            if curve['dev_loss']<best: best=curve['dev_loss']; torch.save(ckpt,output/'best.pt')
        summaries.append(dict(method=method,seed=17,final=curve,best_dev_loss=best,checkpoint=str(output/'best.pt'),parameters=sum(p.numel() for p in model.parameters()),natural_training_completed=False,revision_loss_has_effect=False))
    with (root/'reports/learning_curves_v4.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    report=dict(scope='micro_overfit_diagnostic',results=summaries,natural_training_status='not_run',revision_loss_ablation_interpretation='No eligible revision labels in this fixture. no_revision_loss is an identity control and cannot establish revision-loss benefit.')
    write(root/'reports/diagnostic_training_v4.json',report); return report
