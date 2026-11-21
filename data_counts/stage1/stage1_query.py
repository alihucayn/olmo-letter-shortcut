"""Stage-1 label counts on infini-gram index v4_olmo-mix-1124_llama (consolidated from the notebook cells as run).

Run history (logic unchanged here): counts were first issued with requests + 6 threads; 95 got HTTP 403 'Forbidden'
(throttling) and were re-issued sequentially; hung TLS handshakes led to the final transport below
(curl subprocess, --max-time 60, subprocess timeout 75 s, up to 6 attempts with backoff). Every attempt is appended to
query_log.jsonl. Tokenizer: hf-internal-testing/llama-tokenizer tokenizer.json (Llama-2 vocab), downloaded to llama_tok/.
"""
import json, subprocess, time, threading
import pandas as pd
from tokenizers import Tokenizer

URL = "https://api.infini-gram.io/"
IDX = "v4_olmo-mix-1124_llama"
T = Tokenizer.from_file("llama_tok/tokenizer.json")
LET, DIG = list("ABCD"), list("1234")
LOGF = open("query_log.jsonl", "a")
lk = threading.Lock()


def enc(s):
    e = T.encode(s, add_special_tokens=False)
    return e.ids, e.tokens


# family id, template, locked stage-1 primary?
FAM = [("T1", "Answer: {X}\n", True), ("T2", "answer is {X}\n", True), ("T3", "answer is {X}.\n", True),
       ("T4", "Answer: ({X})\n", True), ("T1dot", "Answer: {X}.\n", False),
       ("P1", "Answer: {X}", False), ("P1b", "answer: {X}", False), ("P2", "answer is {X}", False),
       ("P2p", "answer is ({X})", False), ("P3a", "correct answer is {X}", False),
       ("P3b", "correct option is {X}", False), ("P4", "Answer: ({X})", False),
       ("P5", "({X})", False), ("P6dot", "\n{X}.", False), ("P6par", "\n{X})", False)]


def variant_ids(s, v):
    """sp: first word carries the SentencePiece space marker (mid-line / doc start);
    nosp: first word without it (after newline or non-space char); nl: string begins with newline token."""
    if v == "sp":
        ids, tk = enc("x " + s); assert ids[0] == 921; return ids[1:], tk[1:]
    if v == "nosp":
        ids, tk = enc("x\n" + s); assert ids[:2] == [921, 13]; return ids[2:], tk[2:]
    if v == "nl":
        ids, tk = enc("x" + s); assert ids[0] == 921; return ids[1:], tk[1:]


def post(payload):
    t0 = time.time()
    cmd = ["curl", "-sS", "--connect-timeout", "20", "--max-time", "60", "-X", "POST", URL,
           "-H", "Content-Type: application/json", "-d", json.dumps({"index": IDX, **payload}), "-w", "\n%{http_code}"]
    p = None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=75)
        out = p.stdout.rsplit("\n", 1); j = json.loads(out[0]); code = int(out[1])
    except subprocess.TimeoutExpired:
        j = {"error": "subprocess timeout 75s"}; code = None
    except Exception as e:
        j = {"error": ((p.stderr or p.stdout) if p else repr(e))[:200]}; code = None
    return j, code, round(time.time() - t0, 2)


def Qs(payload, tag, max_att=6):
    for att in range(max_att):
        j, code, w = post(payload)
        e = {**tag, "attempt": att, "http": code, "wall_s": w, "request": payload, "response": j}
        with lk:
            LOGF.write(json.dumps(e) + "\n"); LOGF.flush()
        if "message" not in j and "error" not in j:
            time.sleep(0.3); return j
        time.sleep(min(15 * (att + 1), 60))
    return j


def build_rows():
    rows = []
    for fam, tmpl, prim in FAM:
        vs = ["nl"] if fam.startswith("P6") else ["sp", "nosp"]
        for X in LET + DIG:
            s = tmpl.format(X=X)
            for v in vs:
                ids, tk = variant_ids(s, v)
                rows.append(dict(family=fam, pattern=s, variant=v, label=X,
                                 label_set="letters" if X in LET else "digits",
                                 token_ids=ids, tokens=tk, primary=prim))
    return rows


def main():
    total = Qs({"query_type": "count", "query_ids": []}, {"step": "index_size"})  # total tokens
    rows = build_rows()
    for r in rows:  # counts by token ids
        j = Qs({"query_type": "count", "query_ids": r["token_ids"]},
               {"family": r["family"], "variant": r["variant"], "label": r["label"]})
        r["count"], r["approx"], r["api_token_ids"] = j.get("count"), j.get("approx"), j.get("token_ids")
    # check: API's own tokenisation of the plain string equals the local sp ids
    mism = []
    for r in [r for r in rows if r["variant"] == "sp"]:
        j = Qs({"query_type": "count", "query": r["pattern"]},
               {"family": r["family"], "variant": "sp(string query)", "label": r["label"]})
        if j.get("token_ids") != r["token_ids"]:
            mism.append((r["family"], r["label"]))
    print("index total tokens", total.get("count"), "string-vs-id mismatches", len(mism))
    # template check for T1 sp: matches preceded by '▁' (29871) and by '▁\xa0' (20246)
    tmpl = {}
    for pre_tok, name in [(29871, "sp,preceded by '▁' (double space)"), (20246, "sp,preceded by nbsp token")]:
        for X in LET + DIG:
            ids = [pre_tok] + variant_ids("Answer: {}\n".format(X), "sp")[0]
            j = Qs({"query_type": "count", "query_ids": ids},
                   {"family": "T1", "variant": name, "label": X, "step": "template_check"})
            tmpl[(pre_tok, X)] = j.get("count")
    json.dump(rows, open("stage1_rows.json", "w"))
    json.dump({f"{k[0]}|{k[1]}": v for k, v in tmpl.items()}, open("stage1_template_counts.json", "w"))
    return rows, tmpl


def summarise(rows, nb, chk):
    """chk: dict (family, label_set) -> hand-check status (written after reading contexts; see stage1_contexts.py).
    nb: nbsp-template counts per label (prefix token 20246)."""
    df = pd.DataFrame(rows)
    df["check_status"] = [chk[(f, s)] for f, s in zip(df.family, df.label_set)]
    df["primary_flag"] = df.family.isin(["T1", "T2", "T3", "T4"])
    df["exact_or_approx"] = df.approx.map({False: "exact", True: "approx"})
    out = df[["family", "pattern", "variant", "label", "label_set", "token_ids", "tokens", "count",
              "exact_or_approx", "check_status", "primary_flag"]].copy()
    out["pattern"] = out.pattern.map(lambda s: s.encode("unicode_escape").decode())
    out["token_ids"] = out.token_ids.map(json.dumps)
    out["tokens"] = out.tokens.map(lambda t: json.dumps(t, ensure_ascii=False))
    out.to_csv("stage1_letter_counts.csv", index=False)

    def agg(fams):
        s = df[df.family.isin(fams)].groupby("label")["count"].sum(); res = {}
        for S, labs in [("letters", LET), ("digits", DIG)]:
            tot = s[labs].sum()
            res[S] = {X: {"count": int(s[X]), "share": round(s[X] / tot, 4)} for X in labs}; res[S]["total"] = int(tot)
        return res
    prim = agg(["T1", "T2", "T3", "T4"])
    loose = agg(["P1", "P1b", "P2", "P2p", "P3b", "P4"])
    pd_ex = {X: int(df[df.family.isin(["T1", "T2", "T3", "T4"]) & (df.label == X)]["count"].sum() - nb[X]) for X in DIG}
    return prim, loose, pd_ex


if __name__ == "__main__":
    main()
