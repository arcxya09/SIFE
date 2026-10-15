#!/bin/bash
# Second scan batch (44 cases) to fill a many-core server: a single SIFE-2D run keeps
# only about 3 cores busy, so throughput comes from running many cases at once.
#   bash examples/ale_scan2.sh [threads_per_run]
T=${1:-4}
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
# E. multimode roughness ensemble (random phases): 0.5 um rms, 10 seeds
for s in 2 3 4 5 6 7 8 9 10 11; do run rough_0.5um_s$s --nth 64 --rough 2:2:32:0.5 --seed $s; done
# F. seeds for 0.25 and 1 um rms
for a in 0.25 1.0; do for s in 2 3 4 5 6; do run rough_${a}um_s$s --nth 64 --rough 2:2:32:$a --seed $s; done; done
# G. single-mode amplitude scan (linear -> nonlinear) for l = 8 and 16
for l in 8 16; do for a in 0.25 0.5 2.0 4.0; do run l${l}_${a}um --nth 64 --pert 2:$l:$a; done; done
# H. inner ice surface: amplitude scan of the roughness and single modes
for a in 0.25 0.5 2.0; do run iceinner_rough_${a}um --nth 64 --rough 0:2:32:$a; done
for l in 4 8 16 32; do run iceinner_l${l}_1um --nth 64 --pert 0:$l:1.0; done
# I. laser illumination non-uniformity: other modes and amplitudes
for a in 0.01 0.02 0.05; do run illumP4_$a --nth 32 --illum 4:$a; done
for a in 0.01 0.02; do run illumP6_$a --nth 32 --illum 6:$a; done
for a in 0.005 0.1; do run illumP2_$a --nth 32 --illum 2:$a; done
# J. combined seeds and a resolution check
run illumP2_0.01_rough_0.5um --nth 64 --illum 2:0.01 --rough 2:2:32:0.5
run rough_0.5um_nth128 --nth 128 --rough 2:2:32:0.5
