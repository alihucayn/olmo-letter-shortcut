"""Sample and inspect matched contexts for stage-1 patterns (consolidated from the notebook cells as run).
Requires stage1_rows.json from stage1_query.py. Uses the same curl transport / logging (Qs) as stage1_query.py.
Inspection was done by reading the printed windows; the per-family verdicts are the CHECK dict below.
"""
import ast, json, random, re, threading
import concurrent.futures as cf
import pandas as pd
from stage1_query import Qs, LET, DIG

lk = threading.Lock()
rows = json.load(open("stage1_rows.json"))


def do_find(r):
    j = Qs({"query_type": "find", "query_ids": r["token_ids"]},
           {"family": r["family"], "variant": r["variant"], "label": r["label"], "step": "find"})
    r["segs"], r["find_cnt"] = j.get("segment_by_shard"), j.get("cnt")


def plan_jobs(rows, seed=0):
    """Per (family, label): pool all variants' suffix-array segments, draw uniform global ranks without replacement;
    6 for A and 1, 3 for other labels."""
    rng = random.Random(seed); jobs = []
    for (fam, X) in sorted({(r["family"], r["label"]) for r in rows}):
        rs = [r for r in rows if r["family"] == fam and r["label"] == X and r.get("segs")]
        pool = [(r, s, a, b) for r in rs for s, (a, b) in enumerate(r["segs"]) if b > a]
        tot = sum(b - a for _, _, a, b in pool)
        n = min(6 if X in ("A", "1") else 3, tot)
        for g in sorted(rng.sample(range(tot), n)) if tot else []:
            for r, s, a, b in pool:
                if g < b - a:
                    jobs.append((r, s, a + g)); break
                g -= (b - a)
    return jobs


ctx = []


def do_doc(job):
    r, s, rank = job
    j = Qs({"query_type": "get_doc_by_rank", "s": s, "rank": rank, "max_disp_len": 100, "query_ids": r["token_ids"]},
           {"family": r["family"], "variant": r["variant"], "label": r["label"], "step": "doc"})
    sp = j.get("spans")
    if isinstance(sp, str):
        sp = ast.literal_eval(sp)
    if not sp:
        return
    mi = [i for i, (t, m) in enumerate(sp) if m is not None][0]
    pre = "".join(t for t, m in sp[:mi]); mt = sp[mi][0]; post_ = "".join(t for t, m in sp[mi + 1:])
    budget = 300 - len(mt) - 4; b = budget // 2
    c = pre[-b:] + "⟦" + mt + "⟧" + post_[:budget - b]
    try:
        path = json.loads(j.get("metadata")).get("path")
    except Exception:
        path = None
    with lk:
        ctx.append(dict(family=r["family"], variant=r["variant"], label=r["label"], label_set=r["label_set"],
                        shard=s, rank=rank, doc_ix=j.get("doc_ix"), source_path=path, context=c,
                        _full=pre[-400:] + "⟦" + mt + "⟧" + post_[:400]))


def opt_cue(row):
    """Heuristic aid only (not used for verdicts): >=3 distinct option markers of the label set in the window."""
    s = row._full.replace("⟦", "").replace("⟧", "")
    lab = "ABCD" if row.label_set == "letters" else "1234"
    return len(set(re.findall(r'(?:^|\n|\s|\()([' + lab + r'])(?:[.):]|\))\s', s))) >= 3


def short(c, n=55):
    i, j = c.index("⟦"), c.index("⟧")
    return c[max(0, i - n):j + 1 + n].replace("\n", "⏎")


def show(C, fams, labels, n=55):
    for f in fams:
        for _, r in C[(C.family == f) & (C.label.isin(labels))].iterrows():
            print(f, r.variant[:2], r.label, "|", short(r.context, n))


# Hand-check verdicts recorded after reading the windows printed by show()
CHECK = {
    ("T1", "letters"): "ok: sampled contexts are MCQ answer keys (exam dumps, quizzes)",
    ("T1", "digits"): "FLAG: sp variant dominated by forum template 'View: N\\xa0 Answer: <n>' (answer count): nbsp-preceded matches 1:17127/19432, 2:3110/5450, 3:541/3001, 4:153/2507; nosp variant mostly MCQ answer keys",
    ("T2", "letters"): "ok: mostly answer statements for options (some forum/discussion use)",
    ("T2", "digits"): "FLAG: numeric answers (math/puzzles), not option labels (L2)",
    ("T3", "letters"): "ok: mostly 'The correct answer is X.' after options",
    ("T3", "digits"): "FLAG: mostly numeric answers; some option numbers",
    ("T4", "letters"): "ok: '(A)'-style MCQ answer keys",
    ("T4", "digits"): "ok: '(1)'-style MCQ answer keys",
    ("T1dot", "letters"): "ok: MCQ answer keys",
    ("T1dot", "digits"): "mixed: answer keys and numeric/enumerated answers",
    ("P1", "letters"): "ok-ish: sampled A contexts answer keys; unterminated so includes 'Answer: A...' prefixes of longer words",
    ("P1", "digits"): "FLAG: mostly numeric values starting with the digit (1905, 1981, 124, 1st)",
    ("P1b", "letters"): "FLAG: mostly word/article continuations ('answer: A headache', 'answer: BMW', 'answer: Coco')",
    ("P1b", "digits"): "FLAG: list numbering '1)' and numbers",
    ("P2", "letters"): "FLAG: about half non-label (word starts 'A LOT', 'AIDS', 'BOTH', 'Daugavpils')",
    ("P2", "digits"): "FLAG: numeric values (1891, 12, 1/3)",
    ("P2p", "letters"): "mixed: mostly labels, some '(A) yes, (B) no' enumerations",
    ("P2p", "digits"): "FLAG: mostly prose enumerations '(1) ... (2) ...'",
    ("P3a", "letters"): "ok: answer labels",
    ("P3a", "digits"): "FLAG: numeric values",
    ("P3b", "letters"): "ok: answer labels (solution pages)",
    ("P3b", "digits"): "FLAG: mostly numeric values",
    ("P4", "letters"): "ok: answer keys",
    ("P4", "digits"): "mixed: answer keys and prose enumerations",
    ("P5", "letters"): "FLAG: mostly figure panels, math, chord sheets; few options",
    ("P5", "digits"): "FLAG: legal/list enumerations",
    ("P6dot", "letters"): "FLAG: mostly interview Q&A turns ('A.' = Answer) plus some option lists",
    ("P6dot", "digits"): "FLAG: list numbering / rankings",
    ("P6par", "letters"): "FLAG: mostly Q&A turns, some option lists",
    ("P6par", "digits"): "FLAG: list numbering",
}


def main():
    with cf.ThreadPoolExecutor(3) as ex:
        list(ex.map(do_find, [r for r in rows if r["count"]]))
    jobs = plan_jobs(rows, seed=0)
    with cf.ThreadPoolExecutor(3) as ex:
        list(ex.map(do_doc, jobs))
    C = pd.DataFrame(ctx)
    C["opt_cue"] = C.apply(opt_cue, axis=1)
    print(C.groupby(["family", "label_set"]).opt_cue.mean().unstack().round(2).T.to_string())
    # windows read for the hand-check (as run)
    show(C, ["T1", "T4"], ["A", "1", "C"])
    show(C, ["T2", "T3"], ["A", "1"])
    show(C[(C.family == "T1") & (C.variant == "sp")], ["T1"], LET + DIG, 45)
    sec = ["P1", "P1b", "P2", "P2p", "P3a", "P3b", "P4", "P5", "P6dot", "P6par", "T1dot"]
    show(C, sec, ["A"], 38); show(C, sec, ["1"], 30)
    for f, ls in [("P1b", list("BCD")), ("P2", list("BCD")), ("T1", list("234")), ("T2", list("BCD")),
                  ("T4", list("234")), ("P4", list("234"))]:
        show(C, [f], ls, 25)
    with open("stage1_context_samples.jsonl", "w") as fh:
        for _, r in C.iterrows():
            d = {k: r[k] for k in ["family", "variant", "label", "label_set", "shard", "rank", "doc_ix", "source_path", "context"]}
            d["shard"], d["rank"] = int(d["shard"]), int(d["rank"]); d["family_check"] = CHECK[(r.family, r.label_set)]
            assert len(d["context"]) <= 300
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
