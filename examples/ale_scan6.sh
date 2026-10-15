#!/bin/bash
# Batch 6 (36 cases, output/scan): more seeds, smaller roughness, extra modes, combined seeds.
#   bash examples/ale_scan6.sh [threads_per_run]
T=${1:-2}
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p output/scan
run() {
    name=$1; shift
    if pgrep -f "output/scan/$name " > /dev/null; then echo "skip $name (running)"; return; fi
    if [ -f output/scan/$name/summary.json ]; then echo "skip $name (done)"; return; fi
    NUMBA_NUM_THREADS=$T setsid -f nohup $PY examples/krf_target_2d.py --out output/scan/$name "$@" \
        > output/scan/$name.log 2>&1 < /dev/null
    echo "started $name"
}
for s in 12 13 14 15 16 17 18 19 20 21; do run rough_0.5um_s$s --nth 64 --rough 2:2:32:0.5 --seed $s; done
for s in 1 2 3 4 5; do run rough_0.05um_s$s --nth 64 --rough 2:2:32:0.05 --seed $s; done
for a in 0.005 0.01 0.02; do run illumP2_${a}_rough_0.25um --nth 64 --illum 2:$a --rough 2:2:32:0.25; done
for l in 6 10 20; do for a in 0.25 0.5 1.0; do run l${l}_${a}um_nth64 --nth 64 --pert 2:$l:$a; done; done
for a in 0.25 0.5; do for s in 2 3 4; do run iceinner_rough_${a}um_s$s --nth 64 --rough 0:2:32:$a --seed $s; done; done
for s in 2 3 4; do run iceabl_rough_0.5um_s$s --nth 64 --rough 1:2:32:0.5 --seed $s; done
