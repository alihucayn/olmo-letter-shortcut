#!/bin/bash
# OLMo-2 1B letter scoring: one checkpoint revision per Slurm array task.
# Submit (from the login node), N = number of manifest rows with selected==1 (or smoke==1 if MODE=smoke):
#   cd $OLMO_WORK
#   N=$(python3 -c "import csv;print(sum(r['selected']=='1' for r in csv.DictReader(open('checkpoint_manifest.csv'))))")
#   sbatch --array=1-$N%8 --export=ALL,MODE=selected,PROMPTS=$PWD/prompts.jsonl,MANIFEST=$PWD/checkpoint_manifest.csv run_array.sbatch
# Each task: resolve its revision -> skip if parquet exists -> download revision to a per-task dir ->
# score -> remove ONLY that per-task weight dir. Preemption (gpuidle REQUEUE) is safe: rerun is idempotent.
#SBATCH --job-name=olmo_letters
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
##SBATCH --exclude=<site-specific node list removed>
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
exec 2>&1
ROOT=$OLMO_WORK
IMG=$OLMO_WORK/images/torch27.sif
VENV=$ROOT/venv
MODE=${MODE:-selected}            # selected | smoke
PROMPTS=${PROMPTS:-$ROOT/prompts.jsonl}
MANIFEST=${MANIFEST:-$ROOT/checkpoint_manifest.csv}
OUTDIR=${OUTDIR:-$ROOT/runs/scores}
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
REV=$(python3 - "$MANIFEST" "$MODE" "$TASK" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r[sys.argv[2]].strip() == "1"]
print(rows[int(sys.argv[3]) - 1]["revision"])
PY
)
OUT=$OUTDIR/${REV}.parquet
echo "task=$TASK rev=$REV host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} $(date -Is)"
if [ -s "$OUT" ]; then echo "SKIP exists $OUT"; exit 0; fi
mkdir -p "$OUTDIR"
WDIR=$ROOT/weights_tmp/${SLURM_ARRAY_JOB_ID}_${TASK}_${REV}
mkdir -p "$WDIR"
cleanup() { case "$WDIR" in "$ROOT"/weights_tmp/*) rm -rf -- "$WDIR";; esac; }
trap cleanup EXIT
run() { apptainer exec --nv -B /scratch "$IMG" env PYTHONNOUSERSITE=1 HF_HOME=$ROOT/hf_home \
        HF_HUB_CACHE=$WDIR/hub OMP_NUM_THREADS=4 "$VENV/bin/python" "$@"; }
t0=$(date +%s)
COMMIT=$(run -c "from huggingface_hub import HfApi, snapshot_download as s
s('allenai/OLMo-2-0425-1B', revision='$REV', local_dir='$WDIR/model', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.bin'])
print(HfApi().model_info('allenai/OLMo-2-0425-1B', revision='$REV').sha)" | tail -1)
echo "commit=$COMMIT"
echo "download_s=$(( $(date +%s) - t0 )) size=$(du -sh "$WDIR/model" | cut -f1)"
run "$ROOT/score_letters.py" --revision "$REV" --model "$WDIR/model" --prompts "$PROMPTS" --out "$OUT" --commit "$COMMIT" --batch-size ${BATCH:-64} --dtype ${DTYPE:-float32}
ls -lh "$OUT"
