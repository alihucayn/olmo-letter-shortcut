import json, sys, csv, os, glob
import pandas as pd, numpy as np, pyarrow.parquet as pq
ROOT = os.environ["OLMO_WORK"]
revs = [r["revision"] for r in csv.DictReader(open(ROOT + "/sweep/checkpoint_manifest.csv")) if r["selected"].strip() == "1"]
ids = [json.loads(l)["prompt_id"] for l in open(ROOT + "/sweep/prompts.jsonl") if l.strip()]
res, bad = {}, []
for rv in revs:
    for pre, suf in (("none", ""), ("eos", "_eos")):
        p = f"{ROOT}/runs/scores/{rv}{suf}.parquet"; k = f"{rv}|{pre}"
        if not os.path.exists(p): res[k] = {"ok": False, "why": "missing"}; bad.append(k); continue
        df = pd.read_parquet(p); meta = json.loads(pq.read_schema(p).metadata[b"olmo_letters"])
        num = df.select_dtypes(include=[np.number])
        r = {"rows": len(df), "unique_ids": int(df.prompt_id.nunique()), "ids_match_prompts": set(df.prompt_id) == set(ids),
             "nonfinite": int((~np.isfinite(num.to_numpy(dtype=float))).sum()), "meta_prepend_eos": meta["prepend_eos"],
             "dtype": meta["dtype"], "rev_col_ok": bool((df.revision == rv).all()), "commit": meta.get("commit")}
        r["ok"] = (r["rows"] == 26009 and r["unique_ids"] == 26009 and r["ids_match_prompts"] and r["nonfinite"] == 0
                   and r["meta_prepend_eos"] == (pre == "eos") and r["dtype"] == "float32" and r["rev_col_ok"])
        res[k] = r
        if not r["ok"]: bad.append(k)
    hp = f"{ROOT}/runs/health/health_{rv}.json"
    if not os.path.exists(hp): res[rv + "|health"] = {"ok": False, "why": "missing"}; bad.append(rv + "|health")
json.dump({"n_revisions": len(revs), "n_bad": len(bad), "bad": bad, "detail": res}, open(ROOT + "/runs/verify.json", "w"), indent=1)
print("VERIFY n_revisions", len(revs), "n_bad", len(bad), bad)
