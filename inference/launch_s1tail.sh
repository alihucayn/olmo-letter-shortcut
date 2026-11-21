#!/bin/bash
#SBATCH --partition=epyc-128
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
set -uo pipefail
exec 2>&1
ROOT=$OLMO_WORK; SW=$ROOT/sweep_s1tail; W=$PWD
IMG=$OLMO_WORK/images/torch27.sif
mkdir -p $SW $ROOT/runs/scores $ROOT/runs/health $ROOT/logs
for f in score_letters.py score_health.py run_s1tail.sbatch prompts.jsonl s1tail.csv; do cp -f $W/$f $SW/$f; done
sha256sum $SW/* > $ROOT/runs/s1tail_input_sha256.txt
SB="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/sbatch"
done_all() { python3 - <<'PY'
import csv, os, sys
R = __import__("os").environ["OLMO_WORK"]
rows = [r["revision"] for r in csv.DictReader(open(R + "/sweep_s1tail/s1tail.csv"))]
miss = [str(i + 1) for i, rv in enumerate(rows) if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in
     (f"{R}/runs/scores/{rv}.parquet", f"{R}/runs/scores/{rv}_eos.parquet", f"{R}/runs/health/health_{rv}.json"))]
print(",".join(miss))
PY
}
for round in 1 2 3 4; do
  M=$(done_all); echo "round=$round missing=[$M] $(date -Is)"; [ -z "$M" ] && break
  JID=$($SB --parsable --array=${M}%6 $SW/run_s1tail.sbatch) || { sleep 120; continue; }
  JID=${JID%%;*}; echo $JID >> $ROOT/runs/array_jobids_s1tail.txt; echo "submitted $JID"
  while true; do S=$(squeue -h -j "$JID" -o %T 2>/dev/null); rc=$?; if [ $rc -eq 0 ] && [ -z "$S" ]; then break; fi; sleep 60; done
  sacct -j "$JID" -X -P -o JobID,State,ElapsedRaw,NodeList,Start,End > $ROOT/runs/sacct_$JID.txt 2>&1 || true
done
echo "final missing=[$(done_all)]"
cd $ROOT/runs
python3 - <<'PY' > tar_list_s1tail.txt
import csv
for r in csv.DictReader(open(__import__("os").environ["OLMO_WORK"] + "/sweep_s1tail/s1tail.csv")):
    for s in ('', '_eos'): print('scores/' + r['revision'] + s + '.parquet')
PY
tar -cf raw_parquets_s1tail.tar -T tar_list_s1tail.txt && ls -l raw_parquets_s1tail.tar
echo "S1TAIL_LAUNCHER_DONE $(date -Is)"
