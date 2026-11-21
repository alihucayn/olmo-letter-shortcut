#!/usr/bin/env python
"""Assemble question_shares_s1tail.parquet, letter_probs_s1tail.csv, health_s1tail.csv (locked spec section (b))."""
import json, csv, glob, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
RAW, HEALTH, PROMPTS, MANIFEST, OUT = sys.argv[1:6]
TAIL = pd.read_csv(sys.argv[6], dtype=str).revision.tolist()   # s1tail.csv
man_all = pd.read_csv(MANIFEST, dtype=str)
man = man_all[man_all.revision.isin(TAIL)].copy()
assert len(man) == 6, len(man)
man["replicate_endpoint"] = 0
pr = pd.DataFrame([json.loads(l) for l in open(PROMPTS)])[["prompt_id", "qid", "condition", "label_set", "gold_pos"]]
S1_FINAL_TOKENS = int(man_all[man_all.revision == "stage1-step1907359-tokens4001B"].tokens_seen.iloc[0])
LET = [f"lp_sp_let_{x}" for x in "ABCD"]; DIG = [f"lp_sp_then_dig_{x}" for x in "1234"]; SL = [f"lp_sp_then_let_{x}" for x in "ABCD"]
def norm(lp):
    p = np.exp(lp.astype("float64")); return p / p.sum(1, keepdims=True)
frames, verif = [], []
for _, m in man.iterrows():
    for pre, suf in (("none", ""), ("eos", "_eos")):
        p = f"{RAW}/{m.revision}{suf}.parquet"
        df = pd.read_parquet(p); meta = json.loads(pq.read_schema(p).metadata[b"olmo_letters"])
        num = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ok = len(df) == 26009 and df.prompt_id.is_unique and np.isfinite(num).all() and meta["prepend_eos"] == (pre == "eos") and meta["dtype"] == "float32"
        verif.append({"revision": m.revision, "prefix": pre, "rows": len(df), "unique": bool(df.prompt_id.is_unique), "finite": bool(np.isfinite(num).all()), "ok": bool(ok)})
        d = df.merge(pr, on="prompt_id", how="left", validate="1:1")
        assert d.qid.notna().all()
        isl = (d.label_set == "letters").to_numpy()
        sh = np.where(isl[:, None], norm(d[LET].to_numpy()), norm(d[DIG].to_numpy()))
        sl = norm(d[SL].to_numpy())
        mass = np.where(isl, d.mass_sp_let, d.mass_sp_dig_joint)
        g = d.gold_pos.to_numpy(dtype=float); has = ~np.isnan(g); gi = np.where(has, g, 0).astype(int)
        gold_share = np.where(has, sh[np.arange(len(d)), gi], np.nan)
        acc = np.where(has, (sh.argmax(1) == gi).astype(float), np.nan)
        t = pd.DataFrame({"qid": d.qid, "condition": d.condition, "label_set": d.label_set, "mass": mass, "gold_share": gold_share, "acc": acc})
        for k in range(4): t[f"sh{k}"] = sh[:, k]; t[f"sl{k}"] = sl[:, k]
        q = t.groupby(["condition", "label_set", "qid"], sort=True).agg(
            **{f"sh{k}": (f"sh{k}", "mean") for k in range(4)}, **{f"sl{k}": (f"sl{k}", "mean") for k in range(4)},
            gold_share=("gold_share", "mean"), acc=("acc", "mean"), mass=("mass", "mean"), n_prompts=("mass", "size")).reset_index()
        q.insert(0, "prefix", pre); q.insert(0, "revision", m.revision)
        frames.append(q)
Q = pd.concat(frames, ignore_index=True)
Q = Q.merge(man[["revision", "replicate_endpoint"]], on="revision"); Q["s1_tail"] = 1
Q.to_parquet(f"{OUT}/question_shares_s1tail.parquet", index=False)
# per-checkpoint (questions weighted equally)
agg = Q.groupby(["revision", "prefix", "condition", "label_set"], sort=False).agg(
    **{f"sh{k}": (f"sh{k}", "mean") for k in range(4)}, **{f"sl{k}": (f"sl{k}", "mean") for k in range(4)},
    mass=("mass", "mean"), accuracy=("acc", "mean"), gold_share=("gold_share", "mean"), n_questions=("qid", "nunique"), n_prompts=("n_prompts", "sum")).reset_index()
mm = man[["revision", "stage", "ingredient", "step", "tokens_seen", "replicate_endpoint"]].copy()
mm["tokens_seen"] = mm.tokens_seen.astype("int64"); mm["step"] = mm.step.astype("int64")
# L7 plotting convention: stage-2 x = stage-1 final tokens + step * 2^21
mm["tokens_seen_x"] = np.where(mm.stage == "2", S1_FINAL_TOKENS + mm.step * 2**21, mm.tokens_seen)
A = agg.merge(mm, on="revision", how="left"); A["s1_tail"] = 1
cols = ["revision", "prefix", "condition", "label_set", "stage", "ingredient", "step", "tokens_seen", "tokens_seen_x", "replicate_endpoint", "s1_tail"] + [f"sh{k}" for k in range(4)] + [f"sl{k}" for k in range(4)] + ["mass", "accuracy", "gold_share", "n_questions", "n_prompts"]
order = {r: i for i, r in enumerate(man.revision)}
A = A[cols].sort_values(["revision", "prefix", "label_set", "condition"], key=lambda s: s.map(order) if s.name == "revision" else s).reset_index(drop=True)
A.to_csv(f"{OUT}/letter_probs_s1tail.csv", index=False)
# health
rows = []
for _, m in mm.iterrows():
    h = json.load(open(f"{HEALTH}/health_{m.revision}.json")); l = h["loading"]; n = h["nll"]
    rows.append({"revision": m.revision, "stage": m.stage, "ingredient": m.ingredient, "step": m.step, "tokens_seen": m.tokens_seen, "replicate_endpoint": m.replicate_endpoint,
        "commit": h["commit"], "config_sha256": h["config_sha256"], "missing_count": l["missing_count"], "missing_first5": ";".join(l["missing_first5"]),
        "unexpected_count": l["unexpected_count"], "unexpected_first5": ";".join(l["unexpected_first5"]), "mismatched_count": l["mismatched_count"], "mismatched_first5": ";".join(l["mismatched_first5"]),
        "error_msgs_count": l["error_msgs_count"], "tie_word_embeddings": l["tie_word_embeddings"], "n_prompts": n["n_prompts"], "n_predicted_tokens": n["n_predicted_tokens"],
        "nll_nats_per_prompt_mean": n["mean_nll_nats_per_prompt_mean"], "nll_nats_token_weighted": n["mean_nll_nats_token_weighted"],
        "batch_check_max_abs_diff": h.get("batch_check", {}).get("max_abs_diff_per_prompt", np.nan)})
H = pd.DataFrame(rows); H["s1_tail"] = 1; H.to_csv(f"{OUT}/health_s1tail.csv", index=False)
pd.DataFrame(verif).to_csv(f"{OUT}/parquet_verification_s1tail.csv", index=False)
print("Q rows", len(Q), "A rows", len(A), "H rows", len(H), "verif ok", sum(v["ok"] for v in verif), "/", len(verif))
