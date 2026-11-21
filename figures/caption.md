# Caption

**Figure. Answer-letter shares and accuracy of OLMo 2 1B (allenai/OLMo-2-0425-1B) across training checkpoints, with letter shares in the training data.**
Condition: letters (A-D), option order shuffled, prompt prefix `none` (supplementary figure: prefix `eos`).
Top row (a, b): share of answers given to each letter A-D (coloured lines, 95% bands); dashed line at 0.25 (uniform). Bottom row (c, d): accuracy; dashed line at 0.25 (chance).
(a, c) Stage 1 (olmo-mix-1124): all 26 stage-1 checkpoints, x = tokens seen, log scale.
(b, d) Stage 2 (dolmino-mix-1124, ingredient 3): the 24 ingredient-3 checkpoints, x = stage-2 tokens seen in billions (step x 2^21 / 1e9), linear.
Right-margin markers (not on the x axis): in (a) stage-1 data letter shares (T1-T4); in (b) stage-2 data letter shares, same-line definition (U_sameline, filled bars, left slot) and the definition matched to the stage-1 strings (U_T14, open boxes, right slot). Within each slot the four letters are staggered left to right (A, B, C, D) so that equal values remain visible; the marker's horizontal position carries no value.
Open circles / squares at the right edge of (b, d): end points (step 23,852) of ingredients 1 and 2 (replicate end points), displaced horizontally from the ingredient-3 end point for legibility; their x position carries no value. The released `main` weights are not drawn (not on this lineage).

**n.** 500 ARC-Challenge questions x 24 option orders per checkpoint (shuffled condition). Every plotted share and accuracy is computed on this set.

**Bands.** Bands are 95% question-bootstrap intervals of the letter share at each checkpoint. They do not include checkpoint-to-checkpoint variability, which is the larger source of variation between neighbouring checkpoints. Accuracy has no band (no accuracy interval was supplied).

**Stage-2 x axis.** x = 0 is the last stage-1 checkpoint (step 1,907,359); this assumes that stage 2 continues from that checkpoint. The stage-2 x axis is therefore not on the same scale as the stage-1 axis (log tokens seen); the break between the panels marks the stage switch (learning rate and data change together).

**Data shares.** Data letter shares pool questions with any number of options (including 2- and 3-option items), whereas the prompts have four options. Stage-2 counts come from the public pool, not the exact 50B-token sample. Data shares are from data_letter_shares.csv; model values from match_metrics.csv; intervals from bootstrap_intervals.csv.
