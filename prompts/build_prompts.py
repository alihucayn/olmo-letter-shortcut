"""Build OLMo answer-letter prompt set. Source: allenai/ai2_arc, config ARC-Challenge, split test. Seed 0."""
import io, json, random, itertools, requests, pandas as pd
URL="https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet"
SRC="allenai/ai2_arc:ARC-Challenge:test"
N=500; SEED=0
df=pd.read_parquet(io.BytesIO(requests.get(URL,timeout=60).content))
rows=[]
for _,r in df.iterrows():
    lab=list(r["choices"]["label"]); txt=list(r["choices"]["text"])
    if len(txt)!=4: continue
    if lab not in (list("ABCD"),list("1234")): continue   # labels must be A-D or 1-4 (in order)
    if r["answerKey"] not in lab: continue
    rows.append(dict(qid=r["id"],q=r["question"].strip(),opts=[t.strip() for t in txt],gold=lab.index(r["answerKey"])))
n_elig=len(rows)
rng=random.Random(SEED)
sel=sorted(rng.sample(range(n_elig),N))   # indices into eligible list (dataset order)
qs=[rows[i] for i in sel]
LS={"letters":"ABCD","digits":"1234"}
def fmt(q,labs,opts):
    return "Question: "+q+"\n"+"\n".join(f"{l}. {o}".rstrip() for l,o in zip(labs,opts))+"\nAnswer:"
P=[]
def add(**k): P.append(k)
perms=list(itertools.permutations(range(4)))
for q in qs:
    for ls,labs in LS.items():
        add(prompt_id=f"{q['qid']}|{ls}|blanked",qid=q["qid"],source=SRC,condition="blanked",label_set=ls,perm="0123",gold_pos=q["gold"],text=fmt(q["q"],labs,[""]*4))
        for p in perms:
            ps="".join(map(str,p)); gp=p.index(q["gold"])
            t=fmt(q["q"],labs,[q["opts"][i] for i in p])
            add(prompt_id=f"{q['qid']}|{ls}|shuffled|{ps}",qid=q["qid"],source=SRC,condition="shuffled",label_set=ls,perm=ps,gold_pos=gp,text=t)
            if ps=="0123":
                add(prompt_id=f"{q['qid']}|{ls}|original",qid=q["qid"],source=SRC,condition="original",label_set=ls,perm=ps,gold_pos=gp,text=t)
# unigram prompts (question-free)
def labeled(ls,head=""):
    return head+"\n".join(f"{l}." for l in LS[ls])+"\nAnswer:"
uni=[("bare","Answer:",None),("question_empty","Question:\nAnswer:",None),("newline_answer","\nAnswer:",None),
     ("labels_only",None,"") ,("question_labels",None,"Question:\n"),("options_labels",None,"Options:\n")]
for name,txt,head in uni:
    if txt is not None:
        add(prompt_id=f"unigram|{name}",qid="none",source="synthetic",condition="unigram",label_set="letters",perm=None,gold_pos=None,text=txt)
    else:
        for ls in LS:
            add(prompt_id=f"unigram|{name}|{ls}",qid="none",source="synthetic",condition="unigram",label_set=ls,perm=None,gold_pos=None,text=labeled(ls,head))
with open("prompts.jsonl","w") as f:
    for p in P: f.write(json.dumps(p)+"\n")
pd.DataFrame([dict(qid=q["qid"],source=SRC,gold_original_pos=q["gold"]) for q in qs]).to_csv("questions.csv",index=False)
print("eligible",n_elig,"total prompts",len(P))
