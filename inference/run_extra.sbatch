#!/bin/bash
# OLMo-2 1B sweep: one revision per array task; one download; (a) no-prefix scoring, (b) --prepend-eos scoring, (c) health record.
# Idempotent per output; per-task weight dir removed at exit. Based on run_array.sbatch (adds EOS pass + health).
#SBATCH --job-name=olmo_extra
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
##SBATCH --exclude=<SITE-SPECIFIC NODE LIST REMOVED>
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
exec 2>&1
ROOT=$OLMO_WORK
SW=$ROOT/sweep_extra
IMG=$OLMO_WORK/images/torch27.sif
VENV=$ROOT/venv
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
REV=$(python3 - $SW/extras.csv "$TASK" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["selected"].strip() == "1"]
print(rows[int(sys.argv[2]) - 1]["revision"])
PY
)
SC=$ROOT/runs/scores; HE=$ROOT/runs/health
mkdir -p $SC $HE
A=$SC/${REV}.parquet; B=$SC/${REV}_eos.parquet; H=$HE/health_${REV}.json
echo "task=$TASK rev=$REV host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} attempt_restart=${SLURM_RESTART_COUNT:-0} $(date -Is)"
if [ -s "$A" ] && [ -s "$B" ] && [ -s "$H" ]; then echo "SKIP all outputs exist"; exit 0; fi
WDIR=$ROOT/weights_tmp/sweepx_${SLURM_ARRAY_JOB_ID}_${TASK}_${REV}
mkdir -p "$WDIR"
cleanup() { case "$WDIR" in "$ROOT"/weights_tmp/sweepx_*) rm -rf -- "$WDIR";; esac; }
trap cleanup EXIT
run() { apptainer exec --nv -B "$OLMO_WORK" "$IMG" env PYTHONNOUSERSITE=1 HF_HOME=$ROOT/hf_home HF_HUB_CACHE=$WDIR/hub OMP_NUM_THREADS=4 "$VENV/bin/python" "$@"; }
t0=$(date +%s)
COMMIT=$(run -c "from huggingface_hub import HfApi, snapshot_download as s
s('allenai/OLMo-2-0425-1B', revision='$REV', local_dir='$WDIR/model', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.bin'])
print(HfApi().model_info('allenai/OLMo-2-0425-1B', revision='$REV').sha)" | tail -1)
echo "commit=$COMMIT download_s=$(( $(date +%s) - t0 )) size=$(du -sh "$WDIR/model" | cut -f1)"
if [ ! -s "$H" ]; then
  EXTRA=""; [ "$REV" = main ] && EXTRA="--unbatched-check"
  run $SW/score_health.py --revision "$REV" --model "$WDIR/model" --prompts $SW/prompts.jsonl --out "$H" --commit "$COMMIT" $EXTRA
fi
if [ ! -s "$A" ]; then
  run $SW/score_letters.py --revision "$REV" --model "$WDIR/model" --prompts $SW/prompts.jsonl --out "$A" --commit "$COMMIT" --batch-size 64 --dtype float32
fi
if [ ! -s "$B" ]; then
  run $SW/score_letters.py --revision "$REV" --model "$WDIR/model" --prompts $SW/prompts.jsonl --out "$B" --commit "$COMMIT" --batch-size 64 --dtype float32 --prepend-eos
fi
ls -lh "$A" "$B" "$H"
echo "TASK_DONE $REV $(date -Is) wall_s=$(( $(date +%s) - t0 ))"
