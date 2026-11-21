#!/usr/bin/env python
"""Generate every number and table used in report.tex from the result files in this repository.

Writes report/generated/numbers.tex (LaTeX macros) and report/generated/tab_*.tex.
Run from anywhere:  python report/make_tables.py
"""
import json, os, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)
OUT = R('report', 'generated'); os.makedirs(OUT, exist_ok=True)
K1 = 'stage1-step1907359-tokens4001B'; KF = 'stage2-ingredient3-step23852-tokens51B'
DEFS = ['U_sameline', 'U_strict', 'U_T14', 'mix_U_sameline']
DNAME = {'U_sameline': 'Same-line (primary)', 'U_strict': 'Strict (as first locked)', 'U_T14': 'Matched to stage-1 strings',
         'mix_U_sameline': 'Mix-weighted same-line'}
# Compute used (Slurm accounting, sum of ElapsedRaw over the array jobs; one A30 GPU per task).
GPU_HOURS = {'sweep': 15.88, 'stage1tail': 2.48}

M = {}
def mac(name, val):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    M[name] = val
f1 = lambda v: f'{v:.1f}'; f2 = lambda v: f'{v:.2f}'; f3 = lambda v: f'{v:.3f}'; ci = lambda v: f'{int(round(v)):,}'
sgn = lambda v, f=f2: ('$-$' + f(-v)) if v < 0 else ('+' + f(v) if v > 0 else f(v))
def tex(s): return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
def table(path, spec, head, rows, extra=''):
    L = [r'\begin{tabular}{' + spec + '}', r'\toprule', ' & '.join(head) + r' \\', r'\midrule']
    L += [' & '.join(str(c) for c in r) + r' \\' for r in rows]
    L += [r'\bottomrule', r'\end{tabular}' + extra]
    open(os.path.join(OUT, path), 'w').write('\n'.join(L) + '\n')

V = pd.read_csv(R('results', 'verdict_locked_L5.csv')); RB = pd.read_csv(R('results', 'verdict_robustness.csv'))
CK = pd.read_csv(R('results', 'shift_checkpoint_means.csv')); CV = pd.read_csv(R('results', 'checkpoint_variability.csv'))
DS = pd.read_csv(R('results', 'data_letter_shares.csv')); MM = pd.read_csv(R('results', 'match_metrics.csv'))
GATE = pd.read_csv(R('results', 'smoke_gate_bootstrap.csv')); EQ = json.load(open(R('results', 'main_vs_final_equivalence.json')))
ID = json.load(open(R('weights_identification', 'ident_results.json')))
MAN = pd.read_csv(R('checkpoints', 'checkpoint_manifest.csv'))
S1 = pd.read_csv(R('data_counts', 'stage1', 'stage1_letter_counts.csv')); S1['label'] = S1.label.astype(str)
S2 = pd.read_csv(R('data_counts', 'stage2', 'stage2_subset_summary_v2.csv')); S2['label'] = S2.label.astype(str)
QS = pd.read_parquet(R('analysis', 'inputs', 'question_shares.parquet'))

# ---------------- checkpoints and prompts
mac('NBranches', ci(len(MAN))); mac('NStageOne', ci(MAN.revision.str.startswith('stage1-').sum())); mac('NStageTwo', ci(MAN.revision.str.startswith('stage2-').sum()))
mac('NIngredients', str(MAN[MAN.revision.str.startswith('stage2-')].ingredient.nunique()))
mac('NSelected', ci(MAN.selected.sum()))
g = MM[(MM.label_set == 'letters') & (MM.condition == 'shuffled')]
gn = g[g.prefix == 'none']; ge = g[g.prefix == 'eos']
s1 = gn[gn.stage == 1].sort_values('step'); s2 = gn[(gn.stage == 2) & (gn.ingredient == 3) & (gn.is_main == 0)].sort_values('step')
mac('NScored', ci(MM.revision.nunique())); mac('NStageOneScored', ci(len(s1))); mac('NStageTwoScored', ci(len(s2)))
mac('NQuestions', ci(QS[QS.condition == 'shuffled'].qid.nunique()))
pr = sum(1 for _ in open(R('prompts', 'prompts.jsonl'))); mac('NPrompts', ci(pr))

# ---------------- locked test
loc = V[(V.endpoint == 'ingredient3 (locked)') & (V.condition == 'shuffled')]
ln = loc[loc.prefix == 'none'].set_index('definition'); le = loc[loc.prefix == 'eos'].set_index('definition')
r0 = ln.loc['U_sameline']; dm = json.loads(r0.dm_pp); dme = json.loads(le.loc['U_sameline'].dm_pp)
for X, v in zip('ABCD', dm): mac('Shift' + X, sgn(v))
for X, v in zip('ABCD', dme): mac('ShiftEos' + X, sgn(v))
mac('ShiftMax', f2(r0.shift_max_pp)); mac('ShiftMaxLo', f2(r0.shift_lo)); mac('ShiftMaxHi', f2(r0.shift_hi)); mac('ShiftMaxLabel', r0.shift_label)
assert (loc.verdict == 'did not match').all()
rows = []
for d in DEFS:
    a, b = ln.loc[d], le.loc[d]
    rows.append([DNAME[d], ' / '.join(sgn(x, f1) for x in json.loads(a.dd_pp)), a.rank_of_24, f'{f2(a.beta)} [{f2(a.beta_lo)}, {f2(a.beta_hi)}]', f2(a.r_dm_dd), b.rank_of_24, f'{f2(b.beta)} [{f2(b.beta_lo)}, {f2(b.beta_hi)}]'])
table('tab_locked.tex', 'lccccc c', [r'Stage-2 data definition', r'$\Delta d$ A/B/C/D (pp)', 'Rank', r'$\beta$ [95\% CI]', r'$r$', 'Rank (EOS)', r'$\beta$ (EOS)'], rows)
for d, nm in zip(DEFS, ['Sameline', 'Strict', 'Matched', 'Mix']):
    mac('Rank' + nm, str(ln.loc[d].rank_of_24)); mac('RankEos' + nm, str(le.loc[d].rank_of_24))
    mac('Beta' + nm, f2(ln.loc[d].beta)); mac('BetaLo' + nm, f2(ln.loc[d].beta_lo)); mac('BetaHi' + nm, f2(ln.loc[d].beta_hi))

# ---------------- robustness
sg = RB[(RB.label_set == 'letters') & RB.baseline.str.startswith('single')]
mac('RankSingleMin', str(int(sg[sg.prefix == 'none'].rank_of_24.min()))); mac('RankSingleMax', str(int(sg[sg.prefix == 'none'].rank_of_24.max())))
mac('RankSingleEosMin', str(int(sg[sg.prefix == 'eos'].rank_of_24.min()))); mac('RankSingleEosMax', str(int(sg[sg.prefix == 'eos'].rank_of_24.max())))
m7 = RB[(RB.label_set == 'letters') & (RB.baseline == 'mean of last 7 stage-1') & (RB.endpoint == 'mean of last 7 stage-2')]
rows = [[r.prefix.replace('none', 'none').replace('eos', 'EOS'), DNAME[r.definition], r.rank_of_24, f'{f2(r.beta)} [{f2(r.beta_lo)}, {f2(r.beta_hi)}]', f2(r.r_dm_dd), r.verdict] for _, r in m7.iterrows()]
table('tab_mean7.tex', 'llcccl', ['Prefix', 'Stage-2 data definition', 'Rank', r'$\beta$ [95\% CI]', r'$r$', 'Rule outcome'], rows)
mac('NMeanMatchedNone', str(int(((m7.prefix == 'none') & (m7.verdict == 'matched')).sum()))); mac('NMeanMatchedEos', str(int(((m7.prefix == 'eos') & (m7.verdict == 'matched')).sum())))
rows = [[r.prefix.replace('eos', 'EOS'), r.label, f3(r.mean_last7_stage1), f3(r.mean_last7_stage2), sgn(r.shift_pp), f2(r.ckpt_se_pp), f2(r.shift_over_se)] for _, r in CK.iterrows()]
table('tab_shift7.tex', 'llccccc', ['Prefix', 'Letter', 'Stage 1 (last 7)', 'Stage 2 (last 7)', 'Shift (pp)', 'SE (pp)', 'Shift / SE'], rows)
ckn = CK[CK.prefix == 'none'].set_index('label'); cke = CK[CK.prefix == 'eos'].set_index('label')
mac('MeanShiftD', sgn(ckn.loc['D'].shift_pp)); mac('MeanShiftSeD', f2(ckn.loc['D'].ckpt_se_pp)); mac('MeanShiftEosD', sgn(cke.loc['D'].shift_pp)); mac('MeanShiftEosSeD', f2(cke.loc['D'].ckpt_se_pp))
cvn = CV[CV.prefix == 'none'].set_index('set'); a1 = cvn.loc['last 7 stage-1']; a2 = cvn.loc['last 7 stage-2 (ingredient 3)']
sd1 = [a1[f'sd_pp_{X}'] for X in 'ABCD']; sd2 = [a2[f'sd_pp_{X}'] for X in 'ABCD']
mac('SdStageOneMin', f2(min(sd1))); mac('SdStageOneMax', f2(max(sd1))); mac('SdStageTwoMin', f2(min(sd2))); mac('SdStageTwoMax', f2(max(sd2)))

# ---------------- question-bootstrap SE at the final checkpoint and the EOS effect (paired)
qids = sorted(QS[QS.condition == 'shuffled'].qid.unique()); nq = len(qids)
IDX = np.random.default_rng(0).integers(0, nq, size=(2000, nq))
qm = lambda rev, pfx: QS[(QS.revision == rev) & (QS.prefix == pfx) & (QS.condition == 'shuffled') & (QS.label_set == 'letters')].set_index('qid').loc[qids][['sh0', 'sh1', 'sh2', 'sh3']].to_numpy()
se = 100 * qm(KF, 'none')[IDX].mean(1).std(0); mac('QSeMin', f1(se.min())); mac('QSeMax', f1(se.max()))
dq = qm(KF, 'eos') - qm(KF, 'none'); bs = dq[IDX].mean(1); m = dq.mean(0); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); j = int(np.abs(m).argmax())
mac('EosEffectMax', f1(100 * abs(m[j]))); mac('EosEffectLo', f1(100 * abs(lo[j]) if m[j] > 0 else 100 * abs(hi[j]))); mac('EosEffectHi', f1(100 * abs(hi[j]) if m[j] > 0 else 100 * abs(lo[j]))); mac('EosEffectLabel', 'ABCD'[j])

# ---------------- gate
gg = GATE[GATE.rev == 's2_final']; mac('GateNExcl', str(int(gg.excl95.sum()))); mac('GateNExclBonf', str(int(gg.excl_bonf.sum())))

# ---------------- trajectory facts
mac('AccStageOneMin', f3(s1.accuracy.min())); mac('AccStageOneMax', f3(s1.accuracy.max()))
mac('GoldStageOneMin', f3(s1.gold_share.min())); mac('GoldStageOneMax', f3(s1.gold_share.max()))
mac('AccStageTwoFirst', f3(s2.accuracy.iloc[0])); mac('AccStageTwoFirstStep', ci(s2.step.iloc[0])); mac('AccStageTwoLast', f3(s2.accuracy.iloc[-1]))
for X, i in zip('ABCD', range(4)):
    mac(f'ShareStageOneMin{X}', f3(s1[f'share_{i}'].min())); mac(f'ShareStageOneMax{X}', f3(s1[f'share_{i}'].max()))
mac('TvStageOneMin', f3(s1.tv_uniform.min())); mac('TvStageOneMax', f3(s1.tv_uniform.max())); mac('TvStageOneArgmax', ci(s1.loc[s1.tv_uniform.idxmax()].step))
mac('TvStageTwoMin', f3(s2.tv_uniform.min())); mac('TvStageTwoMax', f3(s2.tv_uniform.max()))
k1 = gn[gn.revision == K1].iloc[0]; kf = gn[gn.revision == KF].iloc[0]; mn = gn[gn.is_main == 1].iloc[0]; me = ge[ge.is_main == 1].iloc[0]
for nm, r in (('Kone', k1), ('Kf', kf), ('Main', mn), ('MainEos', me)):
    for X, i in zip('ABCD', range(4)): mac(f'Share{nm}{X}', f3(r[f'share_{i}']))
    mac(f'Acc{nm}', f3(r.accuracy))
mac('NllFirst', f2(s1.nll_nats.iloc[0])); mac('NllFirstStep', ci(s1.step.iloc[0])); mac('NllKone', f2(k1.nll_nats)); mac('NllKf', f2(kf.nll_nats)); mac('NllMain', f2(mn.nll_nats))
rows = []
for pfx, gx in (('none', gn), ('EOS', ge)):
    for ing in (3, 1, 2):
        r = gx[gx.revision == KF.replace('ingredient3', f'ingredient{ing}')].iloc[0]; v = np.array([r[f'share_{i}'] for i in range(4)])
        rows.append([pfx, ing, *[f3(x) for x in v], ''.join(np.array(list('ABCD'))[np.argsort(-v)]), f3(r.accuracy)])
table('tab_endpoints.tex', 'llcccccc', ['Prefix', 'Ingredient', 'A', 'B', 'C', 'D', 'Order', 'Accuracy'], rows)
rep = V[(V.endpoint != 'ingredient3 (locked)') & (V.condition == 'shuffled')]; mac('RepMatched', str(int((rep.verdict == 'matched').sum()))); mac('RepCells', str(len(rep)))
bl = V[(V.endpoint == 'ingredient3 (locked)') & (V.condition == 'blanked')]; mac('BlankMatched', str(int((bl.verdict == 'matched').sum()))); mac('BlankCells', str(len(bl)))
blr = bl[bl.prefix == 'none'].set_index('definition'); mac('BlankRanks', ', '.join(str(blr.loc[d].rank_of_24) for d in DEFS))
dg = RB[(RB.label_set == 'digits') & (RB.prefix == 'none') & (RB.baseline == 'last stage-1 checkpoint')].set_index('definition')
mac('DigitShift', ' / '.join(sgn(x, f1) for x in json.loads(dg.iloc[0].dm_pp))); mac('DigitRanks', ', '.join(str(dg.loc[d].rank_of_24) for d in DEFS))
mac('DigitMatched', str(int((RB[RB.label_set == 'digits'].verdict == 'matched').sum()))); mac('DigitCells', str(int((RB.label_set == 'digits').sum())))

# ---------------- main weights
mv = ID['main_vs']; mac('MainTensors', str(ID['n_tensors_main'])); mac('MainParams', ci(ID['n_params_main']))
assert all(mv[k]['n_tensors_exactly_equal'] == 0 for k in mv), 'unexpected equal tensors'
mac('MainRelLtwo', f2(mv['i3']['rel_L2'])); mac('MainEqMean', f2(EQ['mean_abs_diff_all'])); mac('MainEqMax', f1(EQ['max_abs_diff_all_lp']))
mac('MainNHyp', str(len(mv)))

# ---------------- data shares
lab = DS[DS.label_set == 'letters']
rows = []
names = {'T1-T4': 'Stage 1: four terminated strings', 'U_sameline': 'Stage 2: same-line (primary)', 'U_strict': 'Stage 2: strict (as first locked)',
         'U_T14': 'Stage 2: matched to stage-1 strings', 'mix_U_sameline': 'Stage 2: mix-weighted same-line', 'U_token': 'Stage 2: exact \\texttt{Answer: X} (not used)'}
order = ['T1-T4', 'U_sameline', 'U_strict', 'U_T14', 'mix_U_sameline', 'U_token']
for d in order:
    r = lab[lab.definition == d].iloc[0]; rows.append([names[d], f3(r.s0), f3(r.s1), f3(r.s2), f3(r.s3), ci(r.n)])
table('tab_data.tex', 'lccccr', ['Definition', 'A', 'B', 'C', 'D', 'Matches'], rows)
prim = S1[S1.primary_flag.astype(str).str.lower().isin(['1', 'true', 'yes', 'primary'])]
c1 = prim[prim.label_set == 'letters'].groupby('label')['count'].sum()
for X in 'ABCD': mac('CountStageOne' + X, ci(c1[X]))
mac('CountStageOneTotal', ci(c1.sum())); mac('NStageOneQueries', ci(len(S1)))
p2 = lambda sub, pat: S2[(S2.subset == sub) & (S2.pattern == pat) & (S2.label_set == 'letters')].set_index('label')['count']
mk = p2('flan+math (pooled)', 'U_marker'); mac('MarkerCount', ci(mk['A'])); assert mk[['B', 'C', 'D']].sum() == 0
st = p2('flan+math (pooled)', 'U_strict'); mac('StrictTotal', ci(st.sum()))
cov = S2.drop_duplicates('subset').set_index('subset')
fm = cov.loc['flan+math (pooled)']; mac('ScanShards', ci(fm.shards_scanned)); mac('ScanGB', f1(fm.text_bytes_scanned / 1e9)); mac('ScanDocs', ci(fm.docs_scanned))
for sub, nm in (('dclm', 'Dclm'), ('pes2o', 'Pes'), ('stackexchange', 'Se')): mac('Cov' + nm, f1(100 * cov.loc[sub].byte_coverage))
mac('GpuSweep', f2(GPU_HOURS['sweep'])); mac('GpuTail', f2(GPU_HOURS['stage1tail']))

# ---------------- appendix: every scored revision
ee = ge.set_index('revision'); rows = []
for _, r in gn.sort_values(['is_main', 'replicate_endpoint', 'stage', 'step']).iterrows():
    e = ee.loc[r.revision]
    lab_ = 'main' if r.is_main else (f'S1 {int(r.step):,}' if r.stage == 1 else f'S2-i{int(r.ingredient)} {int(r.step):,}')
    rows.append([lab_, f1(r.tokens_seen_x / 1e9), *[f3(r[f'share_{i}']) for i in range(4)], f3(r.accuracy), *[f3(e[f'share_{i}']) for i in range(4)], f3(e.accuracy), f2(r.nll_nats)])
L = [r'\begin{longtable}{lrcccccccccccc}', r'\caption{Letter shares (shuffled condition) and accuracy at every scored revision. S1 = stage 1, S2-i$k$ = stage 2, ingredient $k$; numbers after the label are training steps. Tokens for \texttt{main} are the manifest value. NLL = mean per-token negative log-likelihood (nats) on the 500 original-order prompts.}\label{tab:allrevs}\\',
     r'\toprule', r' & & \multicolumn{5}{c}{No prefix} & \multicolumn{5}{c}{EOS prefix} & \\', r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}',
     r'Revision & Tokens (B) & A & B & C & D & Acc. & A & B & C & D & Acc. & NLL \\', r'\midrule', r'\endfirsthead',
     r'\toprule Revision & Tokens (B) & A & B & C & D & Acc. & A & B & C & D & Acc. & NLL \\ \midrule', r'\endhead', r'\bottomrule', r'\endlastfoot']
L += [' & '.join(str(c) for c in r) + r' \\' for r in rows] + [r'\end{longtable}']
open(os.path.join(OUT, 'tab_allrevs.tex'), 'w').write('\n'.join(L) + '\n')

with open(os.path.join(OUT, 'numbers.tex'), 'w') as f:
    f.write('% Generated by report/make_tables.py from the files in results/, analysis/inputs/, data_counts/ and weights_identification/.\n')
    for k, v in M.items(): f.write('\\newcommand{\\' + k + '}{' + str(v) + '}\n')
print(f'wrote {len(M)} macros and', sorted(x for x in os.listdir(OUT) if x.startswith('tab_')))
