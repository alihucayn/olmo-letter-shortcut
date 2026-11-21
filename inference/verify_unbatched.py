#!/usr/bin/env python
"""Independent check of a score_letters.py parquet: for each prompt, two unbatched, unpadded
full forwards (model(...).logits[0,-1]) on (a) the prompt and (b) the prompt + space token;
token ids re-derived here from the tokenizer, not read from the parquet. Prints max abs diff per
column group, probability sanity checks and the tokenizer check."""
import argparse, json, math, torch, pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--revision", default="main"); ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--device", default="cuda")
a = ap.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM
t = pq.read_table(a.parquet); meta = json.loads(t.schema.metadata[b"olmo_letters"]); df = t.to_pandas()
tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()
e = lambda s: tok.encode(s, add_special_tokens=False)
space = e(" ")[0] if len(e(" ")) == 1 else None
assert space == meta["space_token_id"], (space, meta["space_token_id"])
mine = {}
for L in "ABCD":
    mine[f"lp_sp_let_{L}"] = (0, e(" " + L)); mine[f"lp_ns_let_{L}"] = (0, e(L)); mine[f"lp_sp_then_let_{L}"] = (1, e(L))
for d in "1234":
    mine[f"lp_ns_dig_{d}"] = (0, e(d)); mine[f"lp_sp_then_dig_{d}"] = (1, e(d))
mine["lp_space"] = (0, [space])
for k, (_, v) in mine.items():
    assert len(v) == 1 and v[0] == meta["label_token_ids"].get(k, space), (k, v)
rows = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(a.prompts) if l.strip()}
grp = lambda k: k.rsplit("_", 1)[0]
mx = {}; sums = []; top = [0, 0]
with torch.inference_mode():
    for _, r in df.iterrows():
        ids = tok(rows[r.prompt_id]["text"])["input_ids"]
        lps = [torch.log_softmax(model(input_ids=torch.tensor([x], device=a.device)).logits[0, -1].float(), -1)
               for x in (ids, ids + [space])]
        sums += [float(lp.exp().sum()) for lp in lps]
        top[0] += int(int(lps[0].argmax()) == int(r.top1_id)); top[1] += int(int(lps[1].argmax()) == int(r.top1_id_after_space))
        for k, (pos, v) in mine.items():
            mx[grp(k)] = max(mx.get(grp(k), 0.0), abs(float(lps[pos][v[0]]) - float(r[k])))
lpcols = [c for c in df.columns if c.startswith("lp_")]
P = df[lpcols].applymap(math.exp) if hasattr(df, "applymap") else df[lpcols].map(math.exp)
masses = [c for c in df.columns if c.startswith("mass_")]
out = {"dtype": a.dtype, "n": len(df), "max_abs_diff_by_group": mx, "max_abs_diff_all": max(mx.values()),
       "top1_agree": f"{top[0]}/{len(df)}", "top1_after_space_agree": f"{top[1]}/{len(df)}",
       "fullvocab_prob_sum_range": [min(sums), max(sums)],
       "label_probs_in_0_1": bool(((P >= 0) & (P <= 1)).all().all()),
       "max_group_mass": {c: float(df[c].max()) for c in masses},
       "all_masses_le_1": bool((df[masses] <= 1 + 1e-6).all().all()),
       "mean_mass": {c: round(float(df[c].mean()), 4) for c in masses},
       "tokenizer_check": meta["tokenizer_check"]}
print("VERIFY", json.dumps(out))
