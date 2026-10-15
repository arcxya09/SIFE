#!/bin/bash
# Third scan batch (39 cases), 2 numba threads each (a run uses ~1.5-3 cores).
#   bash examples/ale_scan3.sh [threads_per_run]
T=${1:-2}
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
# K. roughness seeds at 0.1 and 2 um rms
for a in 0.1 2.0; do for s in 2 3 4 5 6; do run rough_${a}um_s$s --nth 64 --rough 2:2:32:$a --seed $s; done; done
# L. single-mode l scans at 0.5 and 2 um
for a in 0.5 2.0; do for l in 2 4 12 24 32; do run l${l}_${a}um --nth 64 --pert 2:$l:$a; done; done
# M. inner ice roughness seeds
for s in 2 3 4 5 6; do run iceinner_rough_1um_s$s --nth 64 --rough 0:2:32:1.0 --seed $s; done
# N. illumination + roughness combined
run illumP2_0.02_rough_0.5um --nth 64 --illum 2:0.02 --rough 2:2:32:0.5
run illumP4_0.02_rough_0.5um --nth 64 --illum 4:0.02 --rough 2:2:32:0.5
# O. ice/ablator interface (interface 1)
for a in 0.5 1.0; do run iceabl_rough_${a}um --nth 64 --rough 1:2:32:$a; done
for l in 8 16; do run iceabl_l${l}_1um --nth 64 --pert 1:$l:1.0; done
# P. full sphere (odd modes)
run full_illumP1_0.01 --full --nth 64 --illum 1:0.01
run full_illumP1_0.02 --full --nth 64 --illum 1:0.02
run full_illumP3_0.02 --full --nth 64 --illum 3:0.02
run full_rough_0.5um --full --nth 128 --rough 2:1:32:0.5
# Q. resolution checks
run l16_1um_nth32 --nth 32 --pert 2:16:1.0
run l16_1um_nth128 --nth 128 --pert 2:16:1.0
run rough_0.5um_nth96 --nth 96 --rough 2:2:32:0.5
run rough_0.5um_zs1.5 --nth 64 --zone-scale 1.5 --rough 2:2:32:0.5
