"""Offline raw-data adapter. Labels only leave this module via private files."""
import collections, json, pathlib, random, re, zipfile
from .audit import digest,write,append,Blocked
from .schema import InferenceExample

def read_rows(p):
    with pathlib.Path(p).open() as f:
        if str(p).endswith('.jsonl'):
            for line in f:
                if line.strip(): yield json.loads(line)
        else: yield from json.load(f)

def document_id(title,text): return digest([title,text])[:32]

def contexts(row,dataset):
    if dataset=='2wiki':
        for title,sents in row['context']:
            text=' '.join(sents); positions=[]; start=0
            for s in sents: positions.append([start,start+len(s)]); start+=len(s)+1
            yield title,text,positions
    else:
        for p in row['paragraphs']: yield p['title'],p['paragraph_text'],[]

def prepare(config):
    root=pathlib.Path(config['paths']['workdir']); data=pathlib.Path(config['paths']['data_root']); raw=data/'raw'; raw.mkdir(parents=True,exist_ok=True)
    for ds,archive in [('2wiki','2wiki.zip'),('musique_ans','musique.zip')]:
        target=raw/ds; target.mkdir(exist_ok=True)
        with zipfile.ZipFile(root/'assets'/archive) as z:
            for info in z.infolist():
                if info.is_dir() or '__MACOSX' in info.filename: continue
                name=pathlib.Path(info.filename).name
                if name.endswith(('.json','.jsonl')):
                    p=target/name
                    if not p.exists():
                        with z.open(info) as src,p.open('wb') as dst:
                            import shutil; shutil.copyfileobj(src,dst)
    split_manifest={}; audit={}; random_seed=config['splits']['split_seed']
    for ds in ['2wiki','musique_ans']:
        directory=raw/ds
        files=sorted(directory.glob('*.json*')); corpus={}; rawcount=0
        corpusfiles=[p for p in files if (p.name in ['train.json','dev.json','test.json'] if ds=='2wiki' else p.name.startswith(('musique_ans_','musique_full_')))]
        for p in corpusfiles:
            for row in read_rows(p):
                for title,text,positions in contexts(row,ds):
                    did=document_id(title,text); rawcount+=1
                    corpus.setdefault(did,dict(doc_id=did,parent_doc_id=did,title=title,text=text,offsets=[0,len(text)],sentence_offsets=positions))
        if not corpus: raise Blocked(f'No corpus from {directory}')
        out=data/'corpus'/f'{ds}.jsonl'; out.parent.mkdir(exist_ok=True)
        with out.with_suffix('.tmp').open('w') as f:
            for doc in sorted(corpus.values(),key=lambda x:x['doc_id']): f.write(json.dumps(doc,ensure_ascii=False)+'\n')
        out.with_suffix('.tmp').replace(out)
        pools={}
        for split in ['train','dev']:
            filename=f'{split}.json' if ds=='2wiki' else f'musique_ans_v1.0_{split}.jsonl'
            pools[split]=list(read_rows(directory/filename))
        def group(row):
            path=row.get('evidences',row.get('question_decomposition',[]))
            return digest([re.sub(r'\W+',' ',row['question'].lower()).strip(),path])
        rng=random.Random(random_seed); rng.shuffle(pools['train']); rng.shuffle(pools['dev'])
        eval_n=config['splits']['natural_eval_questions_per_dataset']; dev_n=config['splits']['dev_questions_per_dataset']; train_n=config['splits']['train_base_questions_2wiki'] if ds=='2wiki' else 0
        # Stratified MuSiQue selection preserves natural hop proportions (rounding <=1 per stratum).
        if ds=='musique_ans':
            strata=collections.defaultdict(list)
            for row in pools['dev']: strata[len(row.get('question_decomposition',[]))].append(row)
            chosen=[]
            for h,rows in sorted(strata.items()): chosen+=rows[:round(eval_n*len(rows)/len(pools['dev']))]
            if len(chosen)<eval_n:
                used={x['id'] for x in chosen}; chosen += [r for r in pools['dev'] if r['id'] not in used][:eval_n-len(chosen)]
            evaluation=chosen[:eval_n]
        else: evaluation=pools['dev'][:eval_n]
        sealed_questions={re.sub(r'\W+',' ',r['question'].lower()).strip() for r in evaluation}
        available=[r for r in pools['train'] if re.sub(r'\W+',' ',r['question'].lower()).strip() not in sealed_questions]
        seen_groups=set(); groups=[]
        for r in available:
            g=group(r)
            if g not in seen_groups: groups.append(r); seen_groups.add(g)
        selected={'eval':evaluation,'dev':groups[:dev_n],'train':groups[dev_n:dev_n+train_n]}
        aliases={}
        aliasfile=directory/'id_aliases.json'
        if aliasfile.exists():
            # Official alias file is JSONL despite .json extension.
            try: aliases=json.loads(aliasfile.read_text())
            except json.JSONDecodeError:
                aliases={a['Q_id']:a for a in (json.loads(s) for s in aliasfile.read_text().splitlines() if s.strip())}
            if isinstance(aliases,list): aliases={a['Q_id']:a for a in aliases}
        split_manifest[ds]={}; missing=0; sentence_mapping_errors=[]
        for split,rows in selected.items():
            public=data/'public'/ds/f'{split}.jsonl'; private=data/'private'/ds/f'{split}.jsonl'; public.parent.mkdir(parents=True,exist_ok=True); private.parent.mkdir(parents=True,exist_ok=True)
            ids=[]
            with public.open('w') as f,private.open('w') as g:
                for row in rows:
                    native=row.get('_id',row.get('id')); qid=digest([ds,native])[:24]; ids.append(qid)
                    f.write(json.dumps(dict(qid=qid,question=row['question'],visible_documents=[]))+'\n')
                    support=[]; facts=[]
                    if ds=='2wiki':
                        bytitle=collections.defaultdict(list)
                        for title,text,positions in contexts(row,ds): bytitle[title].append((text,positions))
                        for title,sentence in row['supporting_facts']:
                            options=bytitle[title]
                            support.extend(document_id(title,t) for t,pos in options)
                            if len(options)!=1 or not 0<=sentence<len(options[0][1]):
                                sentence_mapping_errors.append(dict(qid=qid,title=title,sentence=sentence,paragraph_lengths=[len(pos) for t,pos in options],reason='ambiguous_title' if len(options)!=1 else 'official_sentence_out_of_bounds'))
                                continue
                            text,positions=options[0]; did=document_id(title,text)
                            facts.append(dict(doc_id=did,title=title,sentence=sentence,offsets=positions[sentence],text=text[slice(*positions[sentence])]))
                    else:
                        for p in row['paragraphs']:
                            if p['is_supporting']: support.append(document_id(p['title'],p['paragraph_text']))
                    missing+=sum(x not in corpus for x in set(support))
                    answers=[row['answer']]+row.get('answer_aliases',[])
                    ar=aliases.get(row.get('answer_id'),{}) if isinstance(aliases,dict) else {}
                    if isinstance(ar,dict): answers+=ar.get('aliases',[])+ar.get('demonyms',[])
                    elif isinstance(ar,list): answers+=ar
                    record=dict(qid=qid,answers=list(dict.fromkeys(answers)),supporting_facts=facts,supporting_doc_ids=sorted(set(support)),evidences=row.get('evidences',[]),gold_decomposition=row.get('question_decomposition',[]),grouping_metadata=dict(native_id=native,group_id=group(row),dataset_type=row.get('type'),hop_count=len(row.get('question_decomposition',[])),split=split),raw_record=row)
                    g.write(json.dumps(record,ensure_ascii=False)+'\n')
            split_manifest[ds][split]=dict(qids=ids,public_sha256=digest(public),private_sha256=digest(private),count=len(ids))
        trainq={re.sub(r'\W+',' ',r['question'].lower()).strip() for r in pools['train']}
        devq={re.sub(r'\W+',' ',r['question'].lower()).strip() for r in pools['dev']}
        audit[ds]=dict(raw_paragraph_occurrences=rawcount,unique_documents=len(corpus),duplicate_occurrences=rawcount-len(corpus),official_reference_count=430225 if ds=='2wiki' else 139416,missing_support_mappings=missing,sentence_mapping_errors=sentence_mapping_errors,raw_train_dev_question_overlap=len(trainq&devq),selected_eval_questions_excluded_from_training=True,entity_disjointness_verified=False,shared_subquestions_audit='pending',near_duplicate_audit='pending',corpus_sha256=digest(out),sources=[dict(path=str(p),sha256=digest(p)) for p in corpusfiles])
    write(root/'manifests/splits.json',dict(seed=random_seed,locked_before_outputs=True,datasets=split_manifest)); write(root/'reports/data_audit.json',audit)
    return audit
