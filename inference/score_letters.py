#!/usr/bin/env python
"""Score next-token log-probs of answer-label tokens after each prompt.

One row per prompt: prompt_id, revision;
  at P(.|prompt):          lp_sp_let_A..D (' A'..' D', single tokens), lp_ns_let_A..D, lp_ns_dig_1..4,
                           lp_space (log P(space token | prompt));
  at P(.|prompt + space):  lp_sp_then_dig_1..4, lp_sp_then_let_A..D;
  masses: mass_<group> = sum of the 4 probabilities, mass_sp_dig_joint = P(space)*mass_sp_then_dig;
  top1_id (after prompt), top1_id_after_space.
' 1'..' 4' are two tokens (space, digit) in the OLMo 2 tokenizer; the script asserts this and
fails if any label tokenises unexpectedly. No normalisation. Token ids, tokenizer facts and
run info are stored in the parquet schema metadata (key b'olmo_letters').

Padding: right padding; the score is read at index (n_real_tokens - 1) of each
row, i.e. the next-token distribution after the LAST real token. Causal masking
means real tokens never attend to right-side pads, and position ids of real
tokens are 0..n-1 exactly as in an unbatched pass.
"""
import argparse, json, math, time, os, sys
import torch, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

# Groups read at position P(. | prompt). With-space digits are NOT here: in the OLMo 2
# tokenizer ' 1'..' 4' are two tokens (space, digit), so their first token is identical.
GROUPS = {
    "sp_let": [" A", " B", " C", " D"],
    "ns_let": ["A", "B", "C", "D"],
    "ns_dig": ["1", "2", "3", "4"],
}
# Groups read at position P(. | prompt + space token): the second token of ' 1'..' 4' / ' A'..' D'
# when written as space + bare label.
SPACE_GROUPS = {
    "sp_then_dig": ["1", "2", "3", "4"],
    "sp_then_let": ["A", "B", "C", "D"],
}

def col(group, s):
    return f"lp_{group}_{s.strip()}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True, help="HF branch/revision label")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B",
                    help="HF repo id, or a local directory holding the revision")
    ap.add_argument("--commit", default=None, help="HF commit sha of the revision (recorded in metadata)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prepend-eos", action="store_true",
                    help="prepend tokenizer.eos_token_id to every prompt's ids (default off)")
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers

    local = os.path.isdir(a.model)
    kw = {} if local else {"revision": a.revision}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, **kw)
    dtype = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, **kw).to(a.device).eval()
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_s = time.time() - t0

    # label token ids resolved at run time; fail loudly on unexpected tokenisation
    enc1 = lambda t: tok.encode(t, add_special_tokens=False)
    space = enc1(" 1")[0]
    assert tok.decode([space]) == " ", f"space token {space} decodes to {tok.decode([space])!r}"
    ids, ntok, tokcheck = {}, {}, {}
    for g, labs in GROUPS.items():
        for lab in labs:
            e = enc1(lab)
            if len(e) != 1:
                sys.exit(f"FATAL: label {lab!r} is {len(e)} tokens {e}; expected 1")
            ids[col(g, lab)] = e[0]; ntok[col(g, lab)] = 1
    for g, labs in SPACE_GROUPS.items():
        for lab in labs:
            e = enc1(lab)
            if len(e) != 1:
                sys.exit(f"FATAL: label {lab!r} is {len(e)} tokens {e}; expected 1")
            ids[col(g, lab)] = e[0]; ntok[col(g, lab)] = 1
    for d in "1234":  # ' d' must be exactly [space, d]
        e = enc1(" " + d)
        if e != [space, enc1(d)[0]]:
            sys.exit(f"FATAL: ' {d}' tokenises as {e}; expected [{space}, {enc1(d)[0]}]")
        tokcheck[f"' {d}'"] = e
    for L in "ABCD":
        tokcheck[f"' {L}'"] = enc1(" " + L)
    for t in ["Answer: A", "Answer: B", "Answer: 1", "Answer: 4", "Answer:"]:
        tokcheck[t] = tok(t)["input_ids"]
    pos0_keys = [k for g in GROUPS for k in ids if k.startswith(f"lp_{g}_")]
    pos1_keys = [k for g in SPACE_GROUPS for k in ids if k.startswith(f"lp_{g}_")]
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    for r in rows:
        assert r["text"].endswith("Answer:"), r["prompt_id"]
    enc = [tok(r["text"])["input_ids"] for r in rows]  # tokenizer defaults (special tokens as the tokenizer adds them)
    if a.prepend_eos:
        enc = [[tok.eos_token_id] + e for e in enc]
    order = sorted(range(len(rows)), key=lambda i: -len(enc[i]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    id0 = torch.tensor([ids[k] for k in pos0_keys], device=a.device)
    id1 = torch.tensor([ids[k] for k in pos1_keys], device=a.device)
    out = [None] * len(rows)
    t1 = time.time()
    with torch.inference_mode():
        for b0 in range(0, len(order), a.batch_size):
            bi = order[b0:b0 + a.batch_size]
            # Each row = prompt ids + [space]. Under the causal mask, position n-1 gives
            # P(. | prompt) and position n gives P(. | prompt + space), identical to two
            # separate passes (checked against two unbatched passes in verify_unbatched.py).
            L = max(len(enc[i]) for i in bi) + 1
            x = torch.full((len(bi), L), pad, dtype=torch.long)
            m = torch.zeros((len(bi), L), dtype=torch.long)
            for j, i in enumerate(bi):
                n = len(enc[i])
                x[j, :n + 1] = torch.tensor(enc[i] + [space]); m[j, :n + 1] = 1
            x, m = x.to(a.device), m.to(a.device)
            last = m.sum(1) - 1          # index of the appended space = last real token
            ar = torch.arange(len(bi), device=a.device)
            h = model.model(input_ids=x, attention_mask=m).last_hidden_state
            lp0 = torch.log_softmax(model.lm_head(h[ar, last - 1]).float(), dim=-1)  # after prompt
            lp1 = torch.log_softmax(model.lm_head(h[ar, last]).float(), dim=-1)      # after prompt + space
            s0, s1 = lp0[:, id0].cpu(), lp1[:, id1].cpu()
            sp = lp0[:, space].cpu()
            t0_, t1_ = lp0.argmax(-1).cpu(), lp1.argmax(-1).cpu()
            for j, i in enumerate(bi):
                d = {"prompt_id": rows[i]["prompt_id"], "revision": a.revision,
                     "n_tokens": len(enc[i])}
                d.update(zip(pos0_keys, s0[j].tolist()))
                d["lp_space"] = float(sp[j])
                d.update(zip(pos1_keys, s1[j].tolist()))
                for g, labs in {**GROUPS, **SPACE_GROUPS}.items():
                    d[f"mass_{g}"] = float(sum(math.exp(d[col(g, lab)]) for lab in labs))
                d["mass_sp_dig_joint"] = math.exp(d["lp_space"]) * d["mass_sp_then_dig"]
                d["top1_id"] = int(t0_[j]); d["top1_id_after_space"] = int(t1_[j])
                out[i] = d
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    run_s = time.time() - t1
    df = pd.DataFrame(out)
    meta = {
        "model": a.model, "revision": a.revision,
        "commit": a.commit or getattr(model.config, "_commit_hash", None),
        "prepend_eos": bool(a.prepend_eos), "dtype": a.dtype, "batch_size": a.batch_size, "padding": "right; score at last real token",
        "label_token_ids": ids, "label_n_subtokens": ntok, "space_token_id": space,
        "pos0_columns": pos0_keys, "pos1_columns_after_space": pos1_keys, "tokenizer_check": tokcheck,
        "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id, "pad_token_id_used": pad,
        "first_ids_of_prompt0": enc[0][:3],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu",
        "load_s": round(load_s, 2), "score_s": round(run_s, 3), "n_prompts": len(rows),
        "prompts_per_s": round(len(rows) / run_s, 2),
    }
    t = pa.Table.from_pandas(df, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"olmo_letters": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(t, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    print("SCORE_DONE", json.dumps({k: meta[k] for k in ["revision", "commit", "gpu", "load_s", "score_s", "prompts_per_s", "space_token_id", "label_token_ids", "tokenizer_check"]}))

if __name__ == "__main__":
    main()
