#!/bin/bash
# Priority subset (~70 cases) of the scan, started in groups of 10 one minute apart to
# avoid sudden load steps (su has crashed twice under high load).  Cases resume from
# output/scan/<case>/checkpoint.pkl; running and finished cases are skipped.
#   bash examples/ale_priority.sh [threads_per_run] [group_size] [pause_s]
T=${1:-2}; G=${2:-10}; PAUSE=${3:-60}
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p output/scan
n=0
run() {
    name=$1; shift
    if pgrep -f "output/scan/$name " > /dev/null; then echo "skip $name (running)"; return; fi
    if [ -f output/scan/$name/summary.json ]; then echo "skip $name (done)"; return; fi
    NUMBA_NUM_THREADS=$T setsid -f nohup $PY examples/krf_target_2d.py --out output/scan/$name "$@" \
        >> output/scan/$name.log 2>&1 < /dev/null
    echo "started $name"
    n=$((n + 1))
    if [ $((n % G)) -eq 0 ]; then sleep $PAUSE; fi
}
# controls and l=6
run sym_nth16 --nth 16
run sym_nth64 --nth 64
run l6_1um_nth16 --nth 16 --pert 2:6:1.0
run l6_1um_nth64 --nth 64 --pert 2:6:1.0
run l6_0.25um_nth16 --nth 16 --pert 2:6:0.25
run l6_0.1um_nth16 --nth 16 --pert 2:6:0.1
# single-mode l scan, 1 um and 0.25/0.5 um
for l in 2 4 8 12 16 24 32; do run l${l}_1um --nth 64 --pert 2:$l:1.0; done
for l in 2 4 12 24 32; do run l${l}_0.5um --nth 64 --pert 2:$l:0.5; done
for l in 8 16; do run l${l}_0.25um --nth 64 --pert 2:$l:0.25; run l${l}_0.5um --nth 64 --pert 2:$l:0.5; done
for l in 6 10 20; do run l${l}_0.25um_nth64 --nth 64 --pert 2:$l:0.25; done
# multimode roughness amplitude scan with 3 seeds each
for a in 0.05 ; do for s in 1 2 3; do run rough_${a}um_s$s --nth 64 --rough 2:2:32:$a --seed $s; done; done
run rough_0.1um --nth 64 --rough 2:2:32:0.1
for s in 2 3; do run rough_0.1um_s$s --nth 64 --rough 2:2:32:0.1 --seed $s; done
run rough_0.25um --nth 64 --rough 2:2:32:0.25
for s in 2 3; do run rough_0.25um_s$s --nth 64 --rough 2:2:32:0.25 --seed $s; done
for s in 2 3 4 5 6; do run rough_0.5um_s$s --nth 64 --rough 2:2:32:0.5 --seed $s; done
run rough_1.0um --nth 64 --rough 2:2:32:1.0
for s in 2 3; do run rough_1.0um_s$s --nth 64 --rough 2:2:32:1.0 --seed $s; done
run rough_2.0um --nth 64 --rough 2:2:32:2.0
# inner ice surface and ice/ablator interface (includes the formerly stalled cases)
run iceinner_rough_1um --nth 64 --rough 0:2:32:1.0
run iceinner_rough_0.25um --nth 64 --rough 0:2:32:0.25
run iceinner_rough_0.5um --nth 64 --rough 0:2:32:0.5
run iceinner_rough_2.0um --nth 64 --rough 0:2:32:2.0
for l in 4 8 16 32; do run iceinner_l${l}_1um --nth 64 --pert 0:$l:1.0; done
run iceabl_rough_0.5um --nth 64 --rough 1:2:32:0.5
run iceabl_rough_1.0um --nth 64 --rough 1:2:32:1.0
run iceabl_l8_1um --nth 64 --pert 1:8:1.0
run iceabl_l16_1um --nth 64 --pert 1:16:1.0
# illumination non-uniformity
for a in 0.005 0.01 0.02 0.05 0.1; do run illumP2_$a --nth 32 --illum 2:$a; done
for a in 0.01 0.02 0.05; do run illumP4_$a --nth 32 --illum 4:$a; done
for a in 0.01 0.02; do run illumP6_$a --nth 32 --illum 6:$a; done
run illumP2_0.01_rough_0.5um --nth 64 --illum 2:0.01 --rough 2:2:32:0.5
echo "started $n cases"
