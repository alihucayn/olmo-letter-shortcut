import json, glob, os, random
import pandas as pd, numpy as np
D = os.environ["OLMO_WORK"] + "/dolmino"; OUT = D + '/final_v2'; os.makedirs(OUT, exist_ok=True)
PATS = ['U_strict','U_sameline','U_newline','U_marker','U_token','U_newline_Qdoc','U_T14','T1','T2','T3','T4']
LAB = {'letters': list('ABCD'), 'digits': list('1234')}
SAMPLED = ['U_sameline','U_newline','U_token','U_T14']
recs = {}; nerr = 0; nrec = 0
for f in sorted(glob.glob(D + '/results_v2/*.jsonl')):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        nrec += 1
        if 'error' in r: nerr += 1; continue
        recs.setdefault(r['shard'], r)
trecs = {}; tnerr = 0
for f in sorted(glob.glob(D + '/results_v2T/*.jsonl')):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if 'error' in r: tnerr += 1; continue
        trecs.setdefault(r['shard'], r)
print('T-group shards ok', len(trecs), 'T error records', tnerr, flush=True)
for sh, r in recs.items():
    t = trecs[sh]
    assert t['docs_scanned'] == r['docs_scanned'] and t['text_bytes'] == r['text_bytes'], sh
    r['counts'].update(t['counts']); r['samples'].update(t['samples'])
    r['dl_retries'] += t['dl_retries']
listing = pd.read_csv(D + '/work/subsets_listing.csv'); sel = listing[listing.selected]
missing = sorted(set(sel.path) - set(recs))
print('records', nrec, 'error records', nerr, 'shards ok', len(recs), 'selected', len(sel), 'missing', len(missing), flush=True)
# v1 comparison (U_strict per shard/label)
v1 = {}
for f in glob.glob(D + '/results/c*.jsonl') + glob.glob(D + '/results/retry*.jsonl'):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if 'error' not in r: v1.setdefault(r['shard'], r)
mism = 0; npair = 0; d_docs = 0
for sh, r in recs.items():
    for s, labs in LAB.items():
        for x in labs:
            a = r['counts'].get('U_strict|%s|%s' % (s, x), [0, 0]); b = v1[sh]['counts'].get('U_strict|%s|%s' % (s, x), [0, 0])
            npair += 1; mism += (a[0] != b[0]); d_docs += (a[1] != b[1])
    if r['docs_scanned'] != v1[sh]['docs_scanned'] or r['text_bytes'] != v1[sh]['text_bytes']: mism += 1000000
print('U_strict v1-v2 mismatching (shard,label) pairs', mism, 'of', npair, 'docs mismatches', d_docs, flush=True)
tdiff = 0
for sh, r in recs.items():
    for ss, labs in LAB.items():
        for x in labs:
            a = r['counts'].get('U_T14|%s|%s' % (ss, x), [0, 0])[0]; b = sum(r['counts'].get('T%d|%s|%s' % (i, ss, x), [0, 0])[0] for i in (1, 2, 3, 4))
            tdiff += (a != b)
print('U_T14 != sum(T1..T4) pairs', tdiff, flush=True)
rows = []
for sh, r in recs.items():
    for p in PATS + ['U_strict_residual']:
        for s, labs in LAB.items():
            for x in labs:
                g = lambda q: r['counts'].get('%s|%s|%s' % (q, s, x), [0, 0])
                if p == 'U_strict_residual': c = g('U_strict')[0] - g('U_sameline')[0] - g('U_newline')[0]; dd = np.nan
                else: c, dd = g(p)
                rows.append((r['subset'], sh, p, x, s, c, dd, r['text_bytes'], r['docs_scanned'], r['file_bytes']))
cnt = pd.DataFrame(rows, columns=['subset','shard','pattern','label','label_set','count','docs','bytes','docs_scanned','file_bytes'])
cnt.to_csv(OUT + '/stage2_letter_counts_v2.csv', index=False)
top = listing.drop_duplicates('path').set_index('path'); cnt['top'] = cnt.shard.map(top.dataset_subset)
def groups():
    for sub in sorted(cnt.subset.unique()): yield sub, cnt.subset == sub, listing.scan_subset == sub, ('full' if cnt[cnt.subset==sub].top.iloc[0] in ('flan','math') else 'sample_PL4')
    yield 'math (pooled)', cnt.top == 'math', listing.dataset_subset == 'math', 'full'
    yield 'flan+math (pooled)', cnt.top.isin(['flan','math']), listing.dataset_subset.isin(['flan','math']), 'full'
out = []
for g, m, lm, scope in groups():
    sub = cnt[m]; tot = listing[lm]; sh = sub.drop_duplicates('shard')
    cov_shards = len(sh) / len(tot); cov_bytes = sh.file_bytes.sum() / tot['bytes'].sum()
    for p in PATS + ['U_strict_residual']:
        for s, labs in LAB.items():
            q = sub[(sub.pattern == p) & (sub.label_set == s)]
            piv = q.pivot_table(index='shard', columns='label', values='count', aggfunc='sum').reindex(columns=labs).fillna(0)
            tots = piv.sum(axis=1); csum = piv.sum(axis=0); T = csum.sum()
            sh_share = piv[tots > 0].div(tots[tots > 0], axis=0)
            dsum = q.groupby('label')['docs'].sum(min_count=1)
            for x in labs:
                out.append(dict(subset=g, scope=scope, pattern=p, label_set=s, label=x, count=int(csum[x]), total_label_set=int(T),
                    share=(csum[x] / T if T > 0 else np.nan), docs_with_match=(int(dsum.get(x)) if pd.notna(dsum.get(x)) else np.nan),
                    between_shard_sd_share=(sh_share[x].std(ddof=1) if len(sh_share) > 1 else np.nan), shards_with_matches=int((tots > 0).sum()),
                    shards_scanned=len(sh), shards_total=len(tot), shard_coverage=cov_shards, file_bytes_scanned=int(sh.file_bytes.sum()),
                    file_bytes_total=int(tot['bytes'].sum()), byte_coverage=cov_bytes, text_bytes_scanned=int(sh.bytes.sum()), docs_scanned=int(sh.docs_scanned.sum())))
pd.DataFrame(out).to_csv(OUT + '/stage2_subset_summary_v2.csv', index=False)
rng = random.Random(0); samples = []
fmset = set(listing[listing.dataset_subset.isin(['flan','math'])].path)
fm = [r for r in recs.values() if r['shard'] in fmset]
for p in SAMPLED:
    for s in LAB:
        key = '%s|%s' % (p, s)
        items = [(r, r['samples'][key]['n']) for r in fm if r['samples'][key]['n'] > 0]
        N = sum(n for _, n in items)
        if N == 0: continue
        picks = sorted(rng.sample(range(N), min(50, N))); cum = 0; k = {}; idx = 0
        for r, n in items:
            c = 0
            while idx < len(picks) and picks[idx] < cum + n: c += 1; idx += 1
            cum += n
            if c: k[r['shard']] = c
        for r, n in items:
            c = k.get(r['shard'], 0)
            if not c: continue
            res = list(r['samples'][key]['res']); rng.shuffle(res)
            for e in res[:c]:
                samples.append(dict(pattern=p, label_set=s, subset=r['subset'], shard=r['shard'], label=e['label'], doc_id=e['doc_id'], before=e['before'], match=e['match'], after=e['after'], population_matches=N))
with open(OUT + '/stage2_match_samples_v2.jsonl', 'w') as f:
    for e in samples: f.write(json.dumps(e) + '\n')
json.dump({'records': nrec, 'error_records': nerr, 'shards_ok': len(recs), 'selected': len(sel), 'missing': missing, 'u_strict_mismatch_pairs': int(mism), 'u_strict_docs_mismatch': int(d_docs), 'pairs_checked': npair,
           'u_t14_ne_components_pairs': int(tdiff), 'T_error_records': tnerr, 'T_shards_ok': len(trecs), 'parse_fail': sum(r['parse_fail'] for r in recs.values()), 'no_text_field': sum(r['no_text_field'] for r in recs.values()),
           'dl_retries_total': sum(r['dl_retries'] for r in recs.values()), 'docs_with_Q_line_and_newline_match': sum(r['docs_with_Q_line_and_newline_match'] for r in recs.values())}, open(OUT + '/aggregate_meta_v2.json', 'w'))
print('samples', len(samples))
