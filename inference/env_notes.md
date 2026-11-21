# Inference environment notes (OLMo 2 1B letter scoring on a Slurm GPU cluster)

Built and verified 2026-11-21 (Slurm job 25312571, a GPU node (name omitted), NVIDIA A30 24 GB, Exclusive_Process).

## Paths
- Project dir (all writes): `$OLMO_WORK/`
  - `venv/` — venv, `--system-site-packages` overlay on the existing image
    `$OLMO_WORK/images/torch27.sif` (torch 2.7.1+cu128). Added: transformers 4.50.3,
    huggingface_hub 0.36.2, pyarrow 26.0.1, pandas 3.1.1 (full freeze: `venv_freeze.txt`).
  - `hf_home/` — HF cache (HF_HOME) for ad-hoc runs (revision `main` cached, ~2.8 GB).
  - `weights_tmp/<arrayjob>_<task>_<rev>/` — per-task weight dirs used by `run_array.sbatch`, removed at task exit.
  - `runs/scores/<revision>.parquet` — array outputs; `logs/` — array logs; `test/` — dummy test inputs.
- Model card requirement: transformers >= 4.48 (config.json written by 4.50.3; pinned 4.50.3).

## Activation (exact)
```bash
ROOT=$OLMO_WORK
IMG=$OLMO_WORK/images/torch27.sif
apptainer exec --nv -B "$OLMO_WORK" $IMG env PYTHONNOUSERSITE=1 HF_HOME=$ROOT/hf_home OMP_NUM_THREADS=4 \
    $ROOT/venv/bin/python $ROOT/score_letters.py --revision main --prompts prompts.jsonl --out out.parquet
```
GPU request (no `--partition=gpu`; submit plugin rejects it):
`#SBATCH --partition=gpuidle --gres=gpu:a30:1 --exclude=<site-specific node list removed>`
(gpuidle is preemptible/REQUEUE; the wrapper is idempotent, so requeue is safe).

## Network
- Compute (GPU) node: huggingface.co API 200, PyPI 200 → downloads run inside the job on the GPU node.
- One revision download (snapshot_download, safetensors) = 2.8 GB in 54 s.

## Array run
```bash
cd $OLMO_WORK
N=$(python3 -c "import csv;print(sum(r['selected']=='1' for r in csv.DictReader(open('checkpoint_manifest.csv'))))")
sbatch --array=1-$N%8 --export=ALL,MODE=selected,PROMPTS=$PWD/prompts.jsonl,MANIFEST=$PWD/checkpoint_manifest.csv run_array.sbatch
```
`MODE=smoke` selects rows with smoke==1. Optional env: `DTYPE` (default float32), `BATCH` (default 64), `OUTDIR`.
`sbatch --test-only --array=1-3%8 run_array.sbatch` was accepted (2026-11-21).

## Test results (2026-11-21, Slurm jobs 25312571 and the re-test after the digit-tokenisation fix; one GPU node (name omitted), A30)
- Tokenizer: ' A'..' D' = [362],[426],[356],[423]; ' 1'..' 4' = [220,16..19]; 'Answer: A' -> [16533,25,362];
  'Answer: 1' -> [16533,25,220,16]. No BOS is added by the tokenizer (first ids = prompt text).
- Batched (right-padded, batch 8, 20 prompts) vs independent unbatched forwards, max |Δ logprob|:
  float32 6.8e-5 over all label columns; bfloat16 0.358 (0.165 on ' A'..' D'). Default dtype set to float32.
- Throughput, 1000 short dummy prompts, batch 64: float32 99.5 prompts/s, bfloat16 325 prompts/s.
- Load time: cold (HF download + load) 93 s; warm cache 4.3-6.7 s; wrapper per-task: download 55 s (3.1 GB) + load 57.5 s.
- Wrapper body run as array task 1 in-job: per-task weight dir removed (0 left), second call printed SKIP.
