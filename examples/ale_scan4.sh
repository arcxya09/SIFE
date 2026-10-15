#!/bin/bash
# Batch 4 (19 cases, output/scan_v2): key cases rerun with the HLLE spoke-face fix next to
# strong radial shocks (Options2D.shock_hlle, commit after 3555248), plus symmetric controls.
#   bash examples/ale_scan4.sh [threads_per_run]
T=${1:-2}
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
D=output/scan_v2
mkdir -p $D
run() {
    name=$1; shift
    if pgrep -f "$D/$name " > /dev/null; then echo "skip $name (running)"; return; fi
    if [ -f $D/$name/summary.json ]; then echo "skip $name (done)"; return; fi
    NUMBA_NUM_THREADS=$T setsid -f nohup $PY examples/krf_target_2d.py --out $D/$name "$@" > $D/$name.log 2>&1 < /dev/null
    echo "started $name"
}
run sym_nth16 --nth 16
run sym_nth64 --nth 64
run l6_1um_nth16 --nth 16 --pert 2:6:1.0
run l6_1um_nth64 --nth 64 --pert 2:6:1.0
run l6_0.25um_nth16 --nth 16 --pert 2:6:0.25
run l6_0.1um_nth16 --nth 16 --pert 2:6:0.1
for l in 2 4 8 16 32; do run l${l}_1um --nth 64 --pert 2:$l:1.0; done
for a in 0.1 0.25 0.5 1.0; do run rough_${a}um --nth 64 --rough 2:2:32:$a; done
for a in 0.01 0.02; do run illumP2_$a --nth 32 --illum 2:$a; done
run iceinner_rough_1um --nth 64 --rough 0:2:32:1.0
run rough_0.5um_nth128 --nth 128 --rough 2:2:32:0.5
