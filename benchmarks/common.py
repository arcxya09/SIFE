import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output", "benchmarks")
os.makedirs(OUT, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

def hydro_only(**kw):
    from sife import Options
    o = Options(degeneracy=False, cond_e=False, cond_i=False, exchange=False, radiation=False,
                burn=False, verbose=0, dt_max=1e-6, history_every=50)
    for k, v in kw.items():
        setattr(o, k, v)
    return o
