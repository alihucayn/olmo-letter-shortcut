#!/usr/bin/env python
"""Analysis for 'When does OLMo learn its answer-letter shortcut'.
Implements analysis_spec_locked.md (L1-L8, PL1-PL9). Usage: python analyze.py --inp analysis_inputs --out results
Inputs: question_shares*.parquet, letter_probs*.csv, health*.csv (sweep), stage1_letter_counts.csv (infini-gram),
stage2_subset_summary_v2.csv (dolmino scan), dolmino_50B_mix_table13.csv (OLMo 2 paper, Table 13)."""
import argparse, itertools, re, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

LAB = {'letters': list('ABCD'), 'digits': list('1234')}
DEFS = ['U_sameline', 'U_strict', 'U_T14', 'mix_U_sameline']          # PL6 primary, as locked, PL8 matched, PL7 mix-weighted
K1 = 'stage1-step1907359-tokens4001B'; KF = 'stage2-ingredient3-step23852-tokens51B'
PERMS = list(itertools.permutations(range(4)))
step = lambda r: int(re.search(r'step(\d+)', r).group(1))
tv = lambda a, b: 0.5 * np.abs(a - b).sum()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inp', default='analysis_inputs'); ap.add_argument('--out', default='results')
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); P = lambda n: os.path.join(a.inp, n)
    QS = pd.concat([pd.read_parquet(P('question_shares.parquet')).assign(s1_tail=0), pd.read_parquet(P('question_shares_s1tail.parquet'))], ignore_index=True)
    LP = pd.concat([pd.read_csv(P('letter_probs_by_checkpoint.csv')).assign(s1_tail=0), pd.read_csv(P('letter_probs_s1tail.csv'))], ignore_index=True)
    H = pd.concat([pd.read_csv(P('health_by_checkpoint.csv')), pd.read_csv(P('health_s1tail.csv'))], ignore_index=True)

    # ---- data shares
    s1 = pd.read_csv(P('stage1_letter_counts.csv')); s1['label'] = s1.label.astype(str)
    prim = s1[s1.primary_flag.astype(str).str.lower().isin(['1', 'true', 'yes', 'primary'])]
    d1c = {ls: prim[prim.label_set == ls].groupby('label')['count'].sum().reindex(LAB[ls]) for ls in LAB}
    d1 = {ls: (d1c[ls] / d1c[ls].sum()).to_numpy() for ls in LAB}
    s2 = pd.read_csv(P('stage2_subset_summary_v2.csv')); s2['label'] = s2.label.astype(str)
    cov = s2.groupby('subset').byte_coverage.first(); mix = pd.read_csv(P('dolmino_50B_mix_table13.csv'))
    def cnt2(subset, pattern, ls):
        r = s2[(s2.subset == subset) & (s2.pattern == pattern) & (s2.label_set == ls)].set_index('label')['count'].reindex(LAB[ls])
        assert r.notna().all(), (subset, pattern, ls); return r.to_numpy(float)
    D2, rows = {}, []
    for ls in LAB:
        rows.append(dict(stage=1, definition='T1-T4', label_set=ls, n=d1c[ls].sum(), **{f's{i}': v for i, v in enumerate(d1[ls])}))
        for pat in ('U_sameline', 'U_strict', 'U_T14', 'U_token'):
            c = cnt2('flan+math (pooled)', pat, ls); D2[(pat, ls)] = c / c.sum()
            rows.append(dict(stage=2, definition=pat, label_set=ls, n=c.sum(), **{f's{i}': v for i, v in enumerate(c / c.sum())}))
        c = sum(cnt2(r.subset, 'U_sameline', ls) / cov[r.subset] * (r.source_pct_50B / 100) for _, r in mix.iterrows())   # PL7
        D2[('mix_U_sameline', ls)] = c / c.sum()
        rows.append(dict(stage=2, definition='mix_U_sameline', label_set=ls, n=c.sum(), **{f's{i}': v for i, v in enumerate(c / c.sum())}))
    pd.DataFrame(rows).sort_values(['label_set', 'stage', 'definition']).to_csv(os.path.join(a.out, 'data_letter_shares.csv'), index=False)

    # ---- per-question share matrices
    qids = sorted(QS[QS.condition == 'shuffled'].qid.unique()); nq = len(qids)
    Q = {k: g.set_index('qid').loc[qids][['sh0', 'sh1', 'sh2', 'sh3']].to_numpy()
         for k, g in QS[QS.condition.isin(['shuffled', 'blanked'])].groupby(['revision', 'prefix', 'condition', 'label_set'])}
    IDX = np.random.default_rng(a.seed).integers(0, nq, size=(a.boot, nq))

    def L5(M1, Mf, d1v, d2v, labels):
        M1 = np.mean(M1, axis=0) if isinstance(M1, list) else M1; Mf = np.mean(Mf, axis=0) if isinstance(Mf, list) else Mf
        dq = Mf - M1; dm = dq.mean(0); dd = d2v - d1v; T = float(dm @ dd)
        rank = int((np.array([dm @ dd[list(p)] for p in PERMS]) > T + 1e-15).sum()) + 1
        bs = dq[IDX].mean(1); bb = (bs @ dd) / (dd @ dd); lo, hi = np.percentile(bb, [2.5, 97.5])
        bmx = np.abs(bs).max(1); mlo, mhi = np.percentile(bmx, [2.5, 97.5]); dlo, dhi = np.percentile(bs, [2.5, 97.5], axis=0)
        testable = np.abs(dd).max() >= 0.01
        verdict = 'not testable' if not testable else ('matched' if (rank == 1 and lo > 0) else 'did not match')
        return dict(dm_pp=(100 * dm).round(2).tolist(), dm_lo=(100 * dlo).round(2).tolist(), dm_hi=(100 * dhi).round(2).tolist(), dd_pp=(100 * dd).round(2).tolist(),
                    T=T, rank_of_24=rank, beta=T / (dd @ dd), beta_lo=lo, beta_hi=hi, shift_max_pp=100 * np.abs(dm).max(), shift_lo=100 * mlo, shift_hi=100 * mhi,
                    shift_label=labels[int(np.abs(dm).argmax())], r_dm_dd=float(np.corrcoef(dm, dd)[0, 1]), verdict=verdict)

    # ---- locked L5 (single last stage-1 checkpoint -> ingredient end points)
    V = []
    for end, kf in (('ingredient3 (locked)', KF), ('ingredient1 (replicate)', KF.replace('ingredient3', 'ingredient1')), ('ingredient2 (replicate)', KF.replace('ingredient3', 'ingredient2'))):
        for pfx in ('none', 'eos'):
            for cond in ('shuffled', 'blanked'):
                for df in DEFS:
                    V.append(dict(endpoint=end, prefix=pfx, condition=cond, definition=df, baseline='last stage-1 checkpoint', label_set='letters',
                                  **L5(Q[(K1, pfx, cond, 'letters')], Q[(kf, pfx, cond, 'letters')], d1['letters'], D2[(df, 'letters')], LAB['letters'])))
    pd.DataFrame(V).to_csv(os.path.join(a.out, 'verdict_locked_L5.csv'), index=False)

    # ---- PL9 robustness: end-of-stage-1 and end-of-stage-2 checkpoint sets
    s1revs = sorted([r for r in LP[LP.stage == 1].revision.unique()], key=step); tail = s1revs[-7:]
    end7 = sorted([r for r in LP.revision.unique() if r.startswith('stage2-ingredient3')], key=step)[-7:]
    R = []
    for ls in ('letters', 'digits'):
        for pfx in ('none', 'eos'):
            Mf1 = Q[(KF, pfx, 'shuffled', ls)]; Mf7 = [Q[(r, pfx, 'shuffled', ls)] for r in end7]; M17 = [Q[(r, pfx, 'shuffled', ls)] for r in tail]
            for df in DEFS:
                d2v = D2[(df, ls)]; kw = dict(prefix=pfx, definition=df, label_set=ls, condition='shuffled')
                if ls == 'letters':
                    for r in tail: R.append(dict(baseline=f'single: step {step(r)}', endpoint='last stage-2 ckpt', **kw, **L5(Q[(r, pfx, 'shuffled', ls)], Mf1, d1[ls], d2v, LAB[ls])))
                    R.append(dict(baseline='mean of last 7 stage-1', endpoint='last stage-2 ckpt', **kw, **L5(M17, Mf1, d1[ls], d2v, LAB[ls])))
                else:
                    R.append(dict(baseline='last stage-1 checkpoint', endpoint='last stage-2 ckpt', **kw, **L5(Q[(K1, pfx, 'shuffled', ls)], Mf1, d1[ls], d2v, LAB[ls])))
                R.append(dict(baseline='mean of last 7 stage-1', endpoint='mean of last 7 stage-2', **kw, **L5(M17, Mf7, d1[ls], d2v, LAB[ls])))
    pd.DataFrame(R).to_csv(os.path.join(a.out, 'verdict_robustness.csv'), index=False)
    CK, VAR = [], []
    for pfx in ('none', 'eos'):
        S1 = np.array([Q[(r, pfx, 'shuffled', 'letters')].mean(0) for r in tail]); S2 = np.array([Q[(r, pfx, 'shuffled', 'letters')].mean(0) for r in end7])
        dmv = S2.mean(0) - S1.mean(0); se = np.sqrt(S1.var(0, ddof=1) / 7 + S2.var(0, ddof=1) / 7)
        for i, X in enumerate('ABCD'):
            CK.append(dict(prefix=pfx, label=X, mean_last7_stage1=S1[:, i].mean(), mean_last7_stage2=S2[:, i].mean(), shift_pp=100 * dmv[i], ckpt_se_pp=100 * se[i],
                           shift_over_se=dmv[i] / se[i], sd_stage1_pp=100 * S1[:, i].std(ddof=1), sd_stage2_pp=100 * S2[:, i].std(ddof=1)))
        for nm, S in (('last 7 stage-1', S1), ('last 7 stage-2 (ingredient 3)', S2)):
            VAR.append(dict(prefix=pfx, set=nm, **{f'mean_{X}': S[:, i].mean() for i, X in enumerate('ABCD')}, **{f'sd_pp_{X}': 100 * S[:, i].std(ddof=1) for i, X in enumerate('ABCD')},
                            **{f'min_{X}': S[:, i].min() for i, X in enumerate('ABCD')}, **{f'max_{X}': S[:, i].max() for i, X in enumerate('ABCD')}))
    pd.DataFrame(CK).to_csv(os.path.join(a.out, 'shift_checkpoint_means.csv'), index=False); pd.DataFrame(VAR).to_csv(os.path.join(a.out, 'checkpoint_variability.csv'), index=False)

    # ---- per-checkpoint metrics and bootstrap intervals
    meta = LP.drop_duplicates('revision').set_index('revision')[['stage', 'ingredient', 'step', 'tokens_seen_x', 'replicate_endpoint', 's1_tail']]
    uni = QS[QS.condition == 'unigram'].set_index(['revision', 'prefix', 'label_set'])[['sh0', 'sh1', 'sh2', 'sh3']]
    lpi = LP.set_index(['revision', 'prefix', 'condition', 'label_set']); nll = H.set_index('revision').nll_nats_per_prompt_mean
    BI, MM = [], []
    for (rev, pfx, cond, ls), M in Q.items():
        m = M.mean(0); bs = M[IDX].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for i, X in enumerate(LAB[ls]): BI.append(dict(revision=rev, prefix=pfx, condition=cond, label_set=ls, label=X, share=m[i], lo95=lo[i], hi95=hi[i]))
        u = uni.loc[(rev, pfx, ls)].to_numpy(float); L = lpi.loc[(rev, pfx, cond, ls)]
        row = dict(revision=rev, prefix=pfx, condition=cond, label_set=ls, **meta.loc[rev].to_dict(), is_main=int(rev == 'main'), **{f'share_{i}': m[i] for i in range(4)}, labels=''.join(LAB[ls]),
                   accuracy=L.accuracy, gold_share=L.gold_share, mass=L.mass, nll_nats=nll[rev], tv_uniform=tv(m, np.full(4, .25)), tv_unigram=tv(m, u), tv_stage1_data=tv(m, d1[ls]),
                   pearson_stage1=np.corrcoef(m, d1[ls])[0, 1], spearman_stage1=spearmanr(m, d1[ls])[0])
        for df in DEFS:
            d2v = D2[(df, ls)]; row[f'tv_stage2_{df}'] = tv(m, d2v); row[f'pearson_stage2_{df}'] = np.corrcoef(m, d2v)[0, 1]; row[f'spearman_stage2_{df}'] = spearmanr(m, d2v)[0]
        MM.append(row)
    pd.DataFrame(BI).to_csv(os.path.join(a.out, 'bootstrap_intervals.csv'), index=False)
    pd.DataFrame(MM).sort_values(['label_set', 'prefix', 'condition', 'is_main', 'replicate_endpoint', 'stage', 'step']).to_csv(os.path.join(a.out, 'match_metrics.csv'), index=False)
    loc = pd.DataFrame(V); loc = loc[(loc.endpoint == 'ingredient3 (locked)') & (loc.condition == 'shuffled') & (loc.prefix == 'none')]
    print('locked verdicts:', dict(zip(loc.definition, loc.verdict)), 'ranks:', dict(zip(loc.definition, loc.rank_of_24)))

if __name__ == '__main__':
    main()
