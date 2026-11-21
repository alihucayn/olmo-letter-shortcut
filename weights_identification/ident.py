import json, os, sys, time, struct, itertools, urllib.request, urllib.parse
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
R = "allenai/OLMo-2-0425-1B"; W = sys.argv[1]
REVS = {"main": "main", "i1": "stage2-ingredient1-step23852-tokens51B", "i2": "stage2-ingredient2-step23852-tokens51B",
        "i3": "stage2-ingredient3-step23852-tokens51B", "s1": "stage1-step1907359-tokens4001B"}
out = {"revisions": REVS}
paths = {}
for k, rev in REVS.items():
    t = time.time()
    paths[k] = snapshot_download(R, revision=rev, local_dir=f"{W}/{k}", allow_patterns=["*.safetensors", "*.json"])
    print("dl", k, round(time.time() - t), "s", flush=True)
def handles(k):
    idx = json.load(open(f"{paths[k]}/model.safetensors.index.json"))["weight_map"]
    fs = {f: safe_open(f"{paths[k]}/{f}", "pt") for f in set(idx.values())}
    return idx, fs
H = {k: handles(k) for k in REVS}
meta = {k: {n: (str(H[k][1][f].get_slice(n).get_dtype()), list(H[k][1][f].get_slice(n).get_shape())) for n, f in H[k][0].items()} for k in REVS}
keys = sorted(meta["main"])
out["n_tensors_main"] = len(keys); out["n_params_main"] = sum(int(torch.tensor(s).prod()) for _, s in meta["main"].values())
out["dtypes_main"] = sorted({d for d, _ in meta["main"].values()})
hyp = {"i1": ["i1"], "i2": ["i2"], "i3": ["i3"], "s1": ["s1"], "avg_i1_i2_i3": ["i1", "i2", "i3"],
       "avg_i1_i2": ["i1", "i2"], "avg_i1_i3": ["i1", "i3"], "avg_i2_i3": ["i2", "i3"]}
acc = {h: {"max": 0.0, "sum_abs": 0.0, "sq": 0.0, "ref_sq": 0.0, "n": 0, "n_tensors_exact": 0, "per_tensor_max": []} for h in hyp}
for k in ["i1", "i2", "i3", "s1"]:
    out[f"same_keys_shapes_dtypes_main_vs_{k}"] = {"keys": set(meta[k]) == set(meta["main"]),
        "shapes": all(meta[k].get(n, (0, 0))[1] == meta["main"][n][1] for n in keys),
        "dtypes": all(meta[k].get(n, (0, 0))[0] == meta["main"][n][0] for n in keys)}
get = lambda k, n: H[k][1][H[k][0][n]].get_tensor(n).to(torch.float64)
for n in keys:
    m = get("main", n); cache = {k: get(k, n) for k in ["i1", "i2", "i3", "s1"]}
    for h, parts in hyp.items():
        ref = sum(cache[p] for p in parts) / len(parts)
        d = (m - ref).abs(); a = acc[h]
        mx = float(d.max()); a["max"] = max(a["max"], mx); a["sum_abs"] += float(d.sum()); a["sq"] += float((d * d).sum())
        a["ref_sq"] += float((ref * ref).sum()); a["n"] += d.numel(); a["n_tensors_exact"] += int(mx == 0.0)
        a["per_tensor_max"].append((mx, n))
res = {}
for h, a in acc.items():
    a["per_tensor_max"].sort(reverse=True)
    res[h] = {"max_abs_diff": a["max"], "mean_abs_diff": a["sum_abs"] / a["n"], "rel_L2": (a["sq"] / a["ref_sq"]) ** 0.5,
              "n_tensors_exactly_equal": a["n_tensors_exact"], "n_tensors": len(keys),
              "top3_tensors_by_max_diff": [[n, v] for v, n in a["per_tensor_max"][:3]],
              "min_tensor_max_diff": [a["per_tensor_max"][-1][1], a["per_tensor_max"][-1][0]]}
    print(h, {k: v for k, v in res[h].items() if not k.startswith("top")}, flush=True)
out["main_vs"] = res
# one mid-size tensor across all 24 ingredient-3 branches (+ other ends) via HTTP range reads
T = "model.layers.8.self_attn.k_proj.weight"
mainT = get("main", T)
def rng(url, a, b):
    rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(rq, timeout=60).read()
refs = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{R}/refs", timeout=30))["branches"]
br = sorted([x["name"] for x in refs if x["name"].startswith("stage2-ingredient3")], key=lambda s: int(s.split("step")[1].split("-")[0]))
DT = {"F32": (torch.float32, 4), "BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2)}
one = {}
for b in br:
    u0 = f"https://huggingface.co/{R}/resolve/{urllib.parse.quote(b, safe='')}/"
    wm = json.load(urllib.request.urlopen(u0 + "model.safetensors.index.json", timeout=30))["weight_map"]
    u = u0 + wm[T]; n = struct.unpack("<Q", rng(u, 0, 7))[0]; hd = json.loads(rng(u, 8, 7 + n))[T]
    s, e = hd["data_offsets"]; dt, sz = DT[hd["dtype"]]
    x = torch.frombuffer(bytearray(rng(u, 8 + n + s, 8 + n + e - 1)), dtype=dt).reshape(hd["shape"]).to(torch.float64)
    one[b] = {"max_abs_diff_vs_main": float((x - mainT).abs().max()), "dtype": hd["dtype"]}
    print("T", b, one[b], flush=True)
out["one_tensor_scan"] = {"tensor": T, "shape": list(mainT.shape), "ingredient3_branches": one,
                          "n_branches": len(br), "any_exact_match": [b for b, v in one.items() if v["max_abs_diff_vs_main"] == 0.0]}
json.dump(out, open("ident_results.json", "w"), indent=1)
print("DONE")
