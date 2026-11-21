#!/bin/bash
# CPU launcher: stages sweep files into ROOT, sbatch-chains the GPU array (up to 4 rounds, re-submitting only tasks with missing outputs),
# then verifies parquets and tars them. Nothing is deleted except per-task weight dirs (inside the array wrapper).
#SBATCH --partition=epyc-128
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
set -uo pipefail
exec 2>&1
ROOT=$OLMO_WORK; SW=$ROOT/sweep; W=$PWD
IMG=$OLMO_WORK/images/torch27.sif
mkdir -p $SW $ROOT/runs/scores $ROOT/runs/health $ROOT/logs
for f in score_letters.py score_health.py run_sweep.sbatch verify_parquets.py prompts.jsonl checkpoint_manifest.csv; do cp -f $W/$f $SW/$f; done
for f in $W/*.parquet; do cp -n $f $ROOT/runs/scores/$(basename $f); done   # smoke parquets: never overwrite
sha256sum $SW/* > $ROOT/runs/sweep_input_sha256.txt
ls -l $ROOT/runs/scores
SB="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/sbatch"
missing() { python3 - <<'PY'
import csv, os
R = __import__("os").environ["OLMO_WORK"]
rows = [r["revision"] for r in csv.DictReader(open(R + "/sweep/checkpoint_manifest.csv")) if r["selected"].strip() == "1"]
m = [str(i + 1) for i, rv in enumerate(rows) if not all(os.path.getsize(p) > 0 if os.path.exists(p) else False for p in
     (f"{R}/runs/scores/{rv}.parquet", f"{R}/runs/scores/{rv}_eos.parquet", f"{R}/runs/health/health_{rv}.json"))]
print(",".join(m))
PY
}

waitarr() { local J=$1; while true; do S=$(squeue -h -j "$J" -o %T 2>/dev/null); rc=$?; if [ $rc -eq 0 ] && [ -z "$S" ]; then break; fi; sleep 60; done
  sacct -j "$J" -X -P -o JobID,State,ElapsedRaw,NodeList,Start,End > $ROOT/runs/sacct_$J.txt 2>&1 || true; }
MAINIDX=$(python3 - <<'PY'
import csv
rows = [r["revision"] for r in csv.DictReader(open(__import__("os").environ["OLMO_WORK"] + "/sweep/checkpoint_manifest.csv")) if r["selected"].strip() == "1"]
print(rows.index("main") + 1)
PY
)
if [ ! -s $ROOT/runs/health/health_main.json ] || [ ! -s $ROOT/runs/scores/main_eos.parquet ]; then
  CJ=$($SB --parsable --array=$MAINIDX $SW/run_sweep.sbatch) || { echo "canary sbatch failed"; exit 1; }
  CJ=${CJ%%;*}; echo "$CJ" >> $ROOT/runs/array_jobids.txt; echo "canary array $CJ task $MAINIDX"
  waitarr $CJ
  if [ ! -s $ROOT/runs/health/health_main.json ] || [ ! -s $ROOT/runs/scores/main_eos.parquet ]; then
    echo "CANARY FAILED - see $ROOT/logs/olmo_sweep_${CJ}_${MAINIDX}.out"; tail -30 $ROOT/logs/olmo_sweep_${CJ}_${MAINIDX}.out; exit 1; fi
  echo "canary ok"
fi
for round in 1 2 3 4; do
  M=$(missing); echo "round=$round missing_tasks=[$M] $(date -Is)"
  [ -z "$M" ] && break
  JID=$($SB --parsable --array=${M}%8 $SW/run_sweep.sbatch) || { echo "sbatch failed"; sleep 120; continue; }
  echo "$JID" >> $ROOT/runs/array_jobids.txt; echo "submitted array $JID"
  while true; do
    S=$(squeue -h -j "${JID%%;*}" -o %T 2>/dev/null); rc=$?
    if [ $rc -eq 0 ] && [ -z "$S" ]; then break; fi
    sleep 60
  done
  echo "array $JID left queue $(date -Is)"
  sacct -j "${JID%%;*}" -X -P -o JobID,State,ElapsedRaw,NodeList,Start,End > $ROOT/runs/sacct_${JID%%;*}.txt 2>&1 || true
done
echo "final missing=[$(missing)]"
apptainer exec -B "$OLMO_WORK" $IMG env PYTHONNOUSERSITE=1 $ROOT/venv/bin/python $SW/verify_parquets.py || echo "verify failed"
cd $ROOT/runs && tar -cf raw_parquets.tar scores/*.parquet && ls -l raw_parquets.tar
tar -cf health_jsons.tar health/*.json
echo "LAUNCHER_DONE $(date -Is)"
