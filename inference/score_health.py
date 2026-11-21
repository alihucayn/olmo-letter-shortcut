#!/usr/bin/env python
"""Checkpoint health record: loading info, config.json sha256, mean per-token NLL on original-order letter prompts.

NLL: teacher-forced, token positions 2..n (token 1 is never predicted; no BOS/EOS prefix added beyond the tokenizer
default), float32, right padding masked, natural log (nats). Prompts: condition=='original' and label_set=='letters'.
Per-prompt NLL = mean over that prompt's n-1 predicted tokens. Headline = mean over prompts of per-prompt NLL;
token-weighted mean also stored. --unbatched-check re-computes every prompt in its own forward pass (no padding)
and stores per-prompt |batched - unbatched|.
"""
import argparse, json, hashlib, os, time, sys
import torch, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model", required=True, help="local directory holding the revision")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--unbatched-check", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg_path = os.path.join(a.model, "config.json")
    cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model, info = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, output_loading_info=True)
    model = model.to(a.device).eval()
    load_s = time.time() - t0
    def names(x):
        return [k[0] if isinstance(k, (tuple, list)) else str(k) for k in x]
    miss, unex, mism = list(info.get("missing_keys", [])), list(info.get("unexpected_keys", [])), list(info.get("mismatched_keys", []))
    loading = {
        "missing_count": len(miss), "missing_first5": names(miss)[:5],
        "unexpected_count": len(unex), "unexpected_first5": names(unex)[:5],
        "mismatched_count": len(mism), "mismatched_first5": names(mism)[:5],
        "error_msgs_count": len(info.get("error_msgs", [])), "error_msgs_first5": [str(e)[:300] for e in info.get("error_msgs", [])[:5]],
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
    }
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    rows = [r for r in rows if r["condition"] == "original" and r["label_set"] == "letters"]
    assert len(rows) == 500, f"expected 500 original/letters prompts, got {len(rows)}"
    enc = [tok(r["text"])["input_ids"] for r in rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.inference_mode()
    def nll_batch(seqs):
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), pad, dtype=torch.long); m = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = 1
        x, m = x.to(a.device), m.to(a.device)
        logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
        tgt = x[:, 1:].masked_fill(m[:, 1:] == 0, -100)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none").view(len(seqs), -1)
        n = (tgt != -100).sum(1)
        return (ce.sum(1) / n).cpu().tolist(), n.cpu().tolist()

    order = sorted(range(len(rows)), key=lambda i: -len(enc[i]))
    per, ntok = [None] * len(rows), [None] * len(rows)
    t1 = time.time()
    for b0 in range(0, len(order), a.batch_size):
        bi = order[b0:b0 + a.batch_size]
        v, n = nll_batch([enc[i] for i in bi])
        for j, i in enumerate(bi):
            per[i], ntok[i] = v[j], n[j]
    nll_s = time.time() - t1
    tot = sum(p * n for p, n in zip(per, ntok))
    rec = {
        "revision": a.revision, "commit": a.commit, "config_sha256": cfg_sha, "loading": loading,
        "nll": {
            "n_prompts": len(rows), "n_predicted_tokens": int(sum(ntok)), "dtype": "float32", "batch_size": a.batch_size,
            "mean_nll_nats_per_prompt_mean": sum(per) / len(per),
            "mean_nll_nats_token_weighted": tot / sum(ntok),
            "prompt_nll_min": min(per), "prompt_nll_max": max(per),
            "n_tokens_min": min(len(e) for e in enc), "n_tokens_max": max(len(e) for e in enc),
            "first_token_ids_prompt0": enc[0][:3], "last_token_id_prompt0": enc[0][-1],
            "prompt_ids": [r["prompt_id"] for r in rows], "per_prompt_nll": per,
        },
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu",
        "load_s": round(load_s, 2), "nll_s": round(nll_s, 2),
    }
    if a.unbatched_check:
        un = [nll_batch([e])[0][0] for e in enc]
        d = [abs(x - y) for x, y in zip(per, un)]
        rec["batch_check"] = {"n_prompts": len(d), "max_abs_diff_per_prompt": max(d), "mean_abs_diff": sum(d) / len(d),
                              "argmax_prompt_id": rows[max(range(len(d)), key=d.__getitem__)]["prompt_id"],
                              "abs_diff_per_prompt": d, "unbatched_mean_nll": sum(un) / len(un)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    print("HEALTH_DONE", a.revision, json.dumps({"loading": {k: v for k, v in loading.items() if k.endswith("count")},
          "nll": rec["nll"]["mean_nll_nats_per_prompt_mean"], "cfg": cfg_sha[:12], "batch_check": rec.get("batch_check", {}).get("max_abs_diff_per_prompt")}))

if __name__ == "__main__":
    main()
