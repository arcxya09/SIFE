#!/bin/bash
# Parameter scan of the 2D ALE KrF target, one detached process per case.
#   bash examples/ale_scan.sh [threads_per_run]
# Logs: output/scan/<case>.log, results: output/scan/<case>/summary.json
T=${1:-14}
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p output/scan
run() {   # name, args...
    name=$1; shift
    if pgrep -f "output/scan/$name " > /dev/null; then echo "skip $name (running)"; return; fi
    if [ -f output/scan/$name/summary.json ]; then echo "skip $name (done)"; return; fi
    NUMBA_NUM_THREADS=$T setsid -f nohup $PY examples/krf_target_2d.py --out output/scan/$name "$@" \
        > output/scan/$name.log 2>&1 < /dev/null
    echo "started $name"
}
# A. single-mode outer-surface perturbation, 1 um: RT growth versus mode number
for l in 2 4 8 12 16 24 32; do run l${l}_1um --nth 64 --pert 2:$l:1.0; done
# B. multimode outer-surface roughness l = 2..32, rms amplitude scan
for a in 0.1 0.25 1.0 2.0; do run rough_${a}um --nth 64 --rough 2:2:32:$a; done
# C. inner ice-surface (gas/ice interface) roughness
run iceinner_rough_1um --nth 64 --rough 0:2:32:1.0
# D. laser P2 illumination non-uniformity
for a in 0.01 0.02 0.05; do run illumP2_$a --nth 32 --illum 2:$a; done
