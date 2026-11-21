# Analysis specification (locked v2, 2026-11-21)

Locked before any model output on real prompts and before any training-data count existed.
The only model run so far is a 20-prompt dummy pipeline test (label shares not inspected).
The amendments in this section SUPERSEDE the draft text (v1)
reproduced below wherever the two conflict. Later changes must be listed in a dated "post-lock changes" section with the reason.

## Lock amendments

L1. Primary data pattern = one union regex, one count per match position (replaces "P1+P1b+P2+P3+P4 pooled",
    which double-counts: 'correct answer is (B)' matches P2, P3 and P4).
    Loose union  U_loose(X):  \b(?:[Aa]nswer|correct (?:option|choice))(?: is)?:?\s?\(?X\)?(?![A-Za-z0-9])
    Strict union U_strict(X): \b(?:[Aa]nswer|correct (?:option|choice))(?: is)?:?\s?\(?X\)?\.?[ \t]*$   (re.MULTILINE)
    PRIMARY = U_strict (label is the last thing on its line; excludes 'Answer: A dog ...' and 'Answer: 2.5').
    U_loose and families P1-P7 are secondary, reported separately and never summed with each other.
    Stage 1 (infini-gram, exact strings): primary = terminated strings 'Answer: X\n', 'answer is X\n', 'answer is X.\n',
    'Answer: (X)\n' and their hand-verified tokenisation variants, summed only after confirming the strings are mutually exclusive;
    unterminated strings are secondary. If the index cannot express a terminator, stage-1 primary falls back to the
    unterminated strings and this is reported as a stage-1/stage-2 definition mismatch.
L2. Digits are a control, not a primary outcome: in math data 'answer is 4' is usually a numeric answer, not an option label.
L3. Prompt prefix: PRIMARY scoring uses the tokenizer default (no BOS/EOS prefix). Sensitivity: an EOS-prefixed pass on the
    final ingredient-3 checkpoint in the smoke run; reported, not used for the gate.
L4. Gate checkpoint = stage2-ingredient3-step23852-tokens51B (last point of the tracked trajectory), NOT 'main'
    (main's LFS hashes differ from that branch; equivalence unverified). 'main' is scored in the smoke run and compared
    prompt by prompt (max |d logprob| over label columns); if > 1e-2 nats, both are reported and the difference is flagged.
    The criterion itself is unchanged from the approved plan: PASS if, for some X in A-D and some condition in
    {blanked, shuffled}, the question-bootstrap 95% CI of the with-space share excludes 0.25. Reported alongside (not gating):
    Bonferroni 99.375% CIs, max_X |share - 0.25| in pp, and TV(m, uniform).
L5. Decision rule (the draft had none). Label set letters, condition shuffled, primary data pattern:
    Let k1 = last stage-1 checkpoint, kf = gate checkpoint, dm = s_kf - s_k1, dd = d_2 - d_1, T = sum_X dm(X) * dd(X).
    Verdict "matched" iff (i) T is the largest of the 24 values obtained by permuting the label assignment of dd
    (exact permutation p = 1/24 = 0.042, the smallest attainable with 4 labels), AND (ii) the paired question-bootstrap
    95% CI of beta = T / sum dd^2 excludes 0 and is positive. Otherwise "did not match".
    If max_X |dd(X)| < 1 pp, the data shift is too small to test and the verdict is "not testable", not "did not match".
    Supporting, not deciding: Spearman rho(s_k1, d_1) and rho(s_kf, d_2); the same statistics for blanked, for digits, and for U_loose.
L6. "Shift by X points" = max_X |dm(X)| in pp; its CI recomputes the max inside every bootstrap replicate
    (selection inside the bootstrap); all four dm(X) are reported.
L7. Stage-2 tokens_seen on the x-axis = stage-1 final tokens + stage-2 step * 2^21 (assumes stage 2 starts from the last
    stage-1 checkpoint); a plotting convention, stated in the caption.
L8. Inference dtype float32 (bf16 batched-vs-unbatched disagreement 0.358 nats in the harness test; fp32 6.8e-5).

---- draft v1 (verbatim) ----

# Analysis specification, draft v1 (written before any model output was inspected for bias)

Status: draft v1, 2026-11-21 (v1: digit share redefined after a tokenizer check). No model output on real prompts existed when this was written; the only
model run was a 20-prompt dummy pipeline test, whose label shares were not used to shape this spec.
Labels: L = {A,B,C,D} (letters), N = {1,2,3,4} (digits). X ranges over one label set at a time.

## (a) Training-data count patterns

Each pattern is counted separately for every X; a pattern family's share vector is
count(X) / sum over the 4 labels of the same set. Families are reported separately and pooled
(pooled = sum of counts over families P1–P6; P7 is reported but excluded from the pool because
it matches enumerations, not answers).

| id | family | exact strings (X = A..D or 1..4) | regex (Python `re`, MULTILINE) |
|---|---|---|---|
| P1 | answer-colon | `Answer: X` | `\bAnswer:\s?X(?![A-Za-z0-9])` |
| P1b | answer-colon, lowercase | `answer: X` | `\banswer:\s?X(?![A-Za-z0-9])` |
| P2 | answer-is | `answer is X`, `answer is (X)` | `\b[Aa]nswer is:?\s?\(?X\)?(?![A-Za-z0-9])` |
| P3 | correct-answer-is | `correct answer is X`, `correct option is X` | `\bcorrect (?:answer\|option\|choice) is:?\s?\(?X\)?(?![A-Za-z0-9])` |
| P4 | parenthesised in answer context | `Answer: (X)`, `answer is (X)` | `\b[Aa]nswer(?: is)?:?\s?\(X\)` |
| P5 | bare parenthesised | `(X)` | `\(X\)` |
| P6 | option at line start | `X.` , `X)` at start of a line | `^X[.)]\s` |
| P7 | option listing (control) | `\nX. ` / `\nX) ` | same as P6 (P6 and P7 are the same surface; P7 = P6 restricted to lines that are followed within 3 lines by the next label) |

Stage 1 (infini-gram API, index `v4_olmo-mix-1124_llama`, `count` query): exact strings only —
`Answer: A`, ` answer is A`, `(A)`, `\nA.` , `\nA)` and the same for B–D and 1–4 (about 8–16
queries per label set). The index uses the Llama-2 tokenizer, so: (i) each query string is checked
by hand on ≥3 returned documents (`find`/`get_doc_by_rank`) to confirm the count is the intended
surface form; (ii) for each string, record whether a leading space or newline changes the
tokenisation and count both variants where it does; (iii) note that infini-gram counts token-sequence
matches, so `Answer: A` also matches the prefix of `Answer: All…` unless followed by a token boundary
— report counts both raw and, where the API allows a following-token query, with a terminator
(`Answer: A\n`, `Answer: A.`).

Stage 2 (dolmino-mix-1124, Flan and math subsets, read shard by shard): the regexes above on the raw
text of every document in the scanned shards. If a random subset of shards is used, record the shard
list, its fraction of the subset's bytes, and the RNG seed. Stage-2 counts are estimates of the
actual 50B-token training sample (it is a subsample of dolmino-mix-1124 and not public in exact form).

Data share for stage s and label set S: d_s(X) = c_s(X) / Σ_{Y∈S} c_s(Y), for the primary family
(P1 + P2 + P3 + P4 pooled = "answer-context" families) and for each family separately.
Primary family is fixed here: answer-context pool (P1, P1b, P2, P3, P4). The others are secondary.

## (b) Model letter share

Tokenisation (OLMo 2 tokenizer, checked at run time and stored in parquet metadata): ' A'..' D' are
single tokens (362, 426, 356, 423); 'A'..'D' = 32..35; '1'..'4' = 16..19; ' 1'..' 4' are TWO tokens
[220, 16..19] (space, digit). 'Answer: A' -> [..., 25, 362] and 'Answer: 1' -> [..., 25, 220, 16],
so in running text a letter after 'Answer:' is the single token ' X', and a digit is space + digit.
P_p(t) = exp(logprob) from score_letters.py; P_p(t | +sp) is read after appending the space token 220.
- Letter share (primary): s_p(X) = P_p(" X") / Σ_{Y∈A-D} P_p(" Y")  (columns lp_sp_let_*).
- Digit share (primary): s_p(d) = P_p(d | +sp) / Σ_{e∈1-4} P_p(e | +sp)  (columns lp_sp_then_dig_*).
  This equals the with-space joint share P(sp)·P(d|+sp) normalised over 1-4, because P(sp) cancels.
  The first-token ' d' share is NOT used: it is 220 for all four digits, i.e. 0.25 by construction.
- Like-for-like check: space-conditioned letter share P_p(X | +sp) normalised over A-D
  (lp_sp_then_let_*), compared with the digit share computed the same way; and no-space shares
  P_p("X") / P_p("d") at the prompt position (lp_ns_let_*, lp_ns_dig_*) as secondary variants.
- Per question q and condition c: s_{q,c}(X) = mean over that question's prompts in condition c
  (blanked: its blanked prompt(s); shuffled: its 24 permutations).
- Per checkpoint: s_c(X) = mean over questions of s_{q,c}(X) (questions weighted equally).
- Label-set mass (diagnostic, per checkpoint): mean over prompts of mass_sp_let (letters) and
  mass_sp_dig_joint = P(sp)·Σ P(d|+sp) (digits); shares from checkpoints with mean mass < 0.05 are
  flagged "low-mass".
- Unigram prior: s_uni(X) from condition "unigram" prompts, averaged over unigram prompts.
- Accuracy (original and shuffled): argmax over the 4 primary-share labels equals gold_pos.
- Primary condition for the matching analysis: shuffled (content present, position balanced over all
  24 orders, so a content effect cancels across positions). Blanked is reported in parallel.

## (c) Gate criterion (go / no-go for the full checkpoint sweep)

At the final checkpoint (`main`), PASS if for at least one label X in {A,B,C,D} and at least one
of the conditions {blanked, shuffled}, the question-bootstrap 95% percentile CI of s_c(X) excludes
0.25 (with-space shares). Report all 8 (label × condition) CIs. 8 comparisons are made; the CIs are
also reported with a Bonferroni-adjusted 99.375% level as a secondary check, but the gate uses 95%.

## (d) Match metrics per checkpoint k

Vectors: m_k = (s_k(X))_{X∈S} for the model (condition shuffled primary, blanked secondary);
d_1, d_2 = stage-1 and stage-2 data shares.
- Pearson r(m_k, d_s) and Spearman ρ(m_k, d_s) over the 4 labels of S, for S = letters, S = digits,
  and pooled (8 values: letters and digits concatenated, each set's shares summing to 1).
  With n = 4 (8 pooled) these correlations are descriptive; report them with bootstrap CIs (e) and
  do not attach parametric p-values.
- Total-variation distance TV(m_k, d_s) = ½ Σ_X |m_k(X) − d_s(X)| for letters and digits; pooled TV =
  mean of the two.
- Also TV(m_k, uniform) (bias magnitude) and TV(m_k, s_uni,k) (distance from the unigram prior).
- Shift metric at the stage-1 → stage-2 switch: let k1 = last stage-1 checkpoint. For each stage-2
  checkpoint k and label X: Δm_k(X) = s_k(X) − s_{k1}(X); Δd(X) = d_2(X) − d_1(X).
  Report Δm_k(X) in percentage points, Pearson r(Δm_k, Δd) over the 4 labels (per set and pooled),
  and the projection β_k = Σ Δm·Δd / Σ Δd² (fraction of the data shift reproduced).
  "Shift by X points" in the cover-letter sentence = max_X |Δm_final(X)| in pp at the final stage-2
  checkpoint, reported with its bootstrap CI and the label it refers to.

## (e) Bootstrap

2000 resamples of questions with replacement (all prompts of a sampled question travel together),
seed 0, percentile 95% intervals. The same resampled question index set is applied to every
checkpoint and condition in a replicate, so differences between checkpoints (Δm) get paired CIs.
Data shares are treated as fixed in the primary analysis; a secondary analysis resamples stage-2
shards (if a shard sample is used) to give a data-side interval.

## Stated limits (to carry into the note)
- Stage-2 data counts are estimates from dolmino-mix-1124 subsets, not the exact training sample.
- Learning rate and data change together at the switch: a shift shows association, not cause.
- infini-gram uses a different tokenizer from OLMo 2; patterns are verified by hand.


---- POST-LOCK CHANGES ----

2026-11-21, after the smoke run (5 revisions x 26,009 prompts + 1 EOS pass) was inspected. Primary definitions, gate and
verdict rule (L1-L8) are unchanged. Additions:

PL1. EOS-prefixed scoring is extended from the gate checkpoint to every swept checkpoint as a parallel SECONDARY series.
     Reason: at the gate checkpoint the prefix moved shuffled letter shares by up to 4.1 pp (paired 95% CI 3.7-4.5, label C)
     and changed the top letter from B to C; that is the size of the bias itself (max |share-0.25| = 4.5 pp). The primary
     stays no-prefix. The L5 statistic is also computed on the EOS series; if the two verdicts differ the result is
     reported as "prefix-dependent".
PL2. Checkpoint health control for every swept revision: loading info (missing / unexpected / mismatched keys) and mean
     per-token NLL on the 500 original-order letter prompts. Reason: the three stage-1 smoke checkpoints have chance accuracy
     (0.2495, 0.2506, 0.2505) and order-insensitive shares (within-question SD across the 24 orders 0.006-0.010 vs
     0.058-0.067 in stage 2); the control separates a format-prior model from a mis-loaded checkpoint.
PL3. 'main' is not the trajectory end point: per-prompt label log-probs differ from stage2-ingredient3-step23852-tokens51B by
     mean 1.24 nats (max 10.1). 'main' is reported as a separate point, off the curve, pending weight-level identification
     (equal to one branch, an average of ingredient end points, or neither).
PL4. Stage-2 data counts: Flan and math subsets are scanned as planned; a seeded sample of the remaining dolmino subsets is
     scanned too and reported separately, to check whether they contribute answer-label matches.

PL5 (2026-11-21, after weight-level identification of 'main'; weights_identification/ident_results.json).
     'main' equals none of: ingredient 1/2/3 end points, last stage-1 checkpoint, any average of ingredient end points
     (0/179 tensors equal; relative L2 ~1.5; cosine ~0 on layer-8 attention weights, including against step 0).
     Limits: full-tensor comparison on 5 revisions only; other repositories not checked; source of 'main' unidentified.
     Consequences: (i) 'main' is reported as a separate point and is not part of any trajectory statistic;
     (ii) ingredients 1 and 2 are not swept in full; (iii) the END POINTS of ingredients 1 and 2 are scored (both prefixes)
     as replicates of the stage-2 shift from the same stage-1 start. They are supporting only: the L5 verdict is still
     computed on ingredient 3; the L5 statistic for ingredients 1 and 2 is reported next to it.

PL6 (2026-11-21, audit of the stage-2 match samples in data_counts/stage2/stage2_match_samples_v1.jsonl).
     Finding: U_strict as locked allows one whitespace character, including a newline, between the cue and the label.
     In the 50-match random sample of U_strict letter matches (Flan+math), 16 are newline-separated and 9 of those are
     'Answer:\n(A).' followed directly by the answer text, i.e. the turn marker of a '(Q). / (A).' Flan template, all
     counted as option A (9 of the 21 A matches in the sample). The 34 same-line matches all read as option labels.
     Second reason, independent of any outcome: the stage-1 primary strings ('Answer: X\n', 'answer is X\n', ...) are
     same-line forms by construction, so U_strict was not like-for-like with stage 1.
     Change: stage-2 PRIMARY := U_sameline(X) = \b(?:[Aa]nswer|correct (?:option|choice))(?: is)?:?[ \t]?\(?X\)?\.?[ \t]*$
     Kept and reported next to it: U_strict as locked. The L5 verdict is computed under both; if they differ the result
     is reported as "pattern-dependent", not as either verdict.
     Secondary: U_newline (cue, newline, label), its turn-marker class [Aa]nswer:\n\(X\)\.[ \t]*\n(?=\S), and
     U_token(X) = \bAnswer: X[ \t]*$ (the exact continuation that is scored).
     State of knowledge when this was written: Flan/math U_strict and U_loose shares and the smoke-run model shares
     (5 revisions) had been seen; no stage-1 count, no full-sweep result and no match statistic (T, beta, rho) existed.

PL7 (2026-11-21). Mix-weighted stage-2 share, SECONDARY. Source: OLMo 2 paper (arXiv 2501.00656), Table 13, 50B mix
     'Source %' column (fraction of each source used): parsed into dolmino_50B_mix_table13.csv.
     d_2,mix(X) is proportional to sum over sources of (source fraction used) x (estimated full-source count of X), where the
     full-source count is the scanned count for fully scanned sources and (sample count / byte coverage) for sampled ones
     (dclm, pes2o, stackexchange: provisional). Assumes the 1B run used the 50B mix (branch names end in tokens51B).
     The primary d_2 stays Flan+math pooled (as locked, with the PL6 pattern); the L5 statistic is also reported with
     d_2,mix. Written before any stage-1 count, full-sweep result or match statistic existed.

PL8 (2026-11-21). Matched-definition stage-2 pattern, SECONDARY: U_T14(X) = (?:Answer: X|answer is X\.?|Answer: \(X\))\n ,
     the regex equivalent of the four stage-1 infini-gram strings (T1-T4). Reason: the stage-1 pattern check and PL6 both
     identify a stage-1 / stage-2 definition mismatch, which acts directly on dd = d_2 - d_1. The L5 statistic is reported
     with d_2 from U_sameline (primary), U_strict (as locked), U_T14 (matched) and the PL7 mix weighting. Headline wording
     rule: the verdict is stated as "matched" or "did not match" only if all four agree; otherwise "pattern-dependent",
     with the four results listed. Written with the stage-1 primary shares known (A 0.2637, B 0.2560, C 0.2550, D 0.2253;
     all counts exact) and before any v2 stage-2 count, full-sweep result or match statistic existed.

PL9 (2026-11-21, after the full sweep was tabulated; no match statistic (T, beta, rho, TV) had been computed).
     Finding: stage-1 letter shares differ widely between the selected checkpoints (shuffled, no prefix: A 0.148-0.664,
     B 0.072-0.384, C 0.086-0.407, D 0.135-0.332), the selected checkpoints near the switch are far apart (2517B, 4000B),
     and L5 takes its baseline from the single last stage-1 checkpoint with a question-only bootstrap.
     Addition: the six stage-1 branches immediately preceding the last one are scored (both prefixes, health record).
     Reported: (i) SD of each letter share over the last seven stage-1 checkpoints and over the last seven stage-2
     checkpoints (checkpoint-level variability); (ii) the L5 statistic recomputed with each of the seven as baseline and
     with the mean of the seven as baseline, and with the mean of the last seven stage-2 checkpoints as end point.
     The locked L5 statistic (single last stage-1 checkpoint, single last ingredient-3 checkpoint) is still reported.
     Headline rule, tightened: "matched" requires the locked statistic to say matched under all four data definitions
     (PL8) AND the seven-checkpoint-mean version to say matched under the primary definition. If the locked statistic
     says matched but the mean version does not, the headline is "not robust to checkpoint choice".
