#!/bin/bash
# Symmetric controls and l=6 amplitude cases (output/scan), complementing batches 1-3.
#   bash examples/ale_scan5.sh [threads_per_run]
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
run sym_nth16 --nth 16
run sym_nth64 --nth 64
run l6_1um_nth16 --nth 16 --pert 2:6:1.0
run l6_1um_nth64 --nth 64 --pert 2:6:1.0
run l6_0.25um_nth16 --nth 16 --pert 2:6:0.25
run l6_0.1um_nth16 --nth 16 --pert 2:6:0.1
