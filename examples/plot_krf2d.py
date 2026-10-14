"""Compare 2D KrF runs: density / ion-temperature maps at peak velocity and at bang
time, the shape of the fuel inner surface, and the gain.

    python examples/plot_krf2d.py output/krf2d_sym output/krf2d_illumP2 output/krf2d_pertl6
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

from sife.animate import _quads


def _map(ax, S, i, L, title, vmax_rho=None):
    z = S["z"][i] * 1e4; R = S["R"][i] * 1e4
    rho = S["rho"][i]; Ti = S["Ti"][i] / 1e3
    qa = _quads(z, -R)[:, :, ::-1]; qb = _quads(z, R)[:, :, ::-1]
    half = bool(S["half"])
    if half:   # mirror the hemisphere to the full meridional plane
        qa = np.concatenate([qa, qa * np.array([1, -1])]); qb = np.concatenate([qb, qb * np.array([1, -1])])
        rho = np.concatenate([rho.ravel(), rho.ravel()]); Ti = np.concatenate([Ti.ravel(), Ti.ravel()])
    vmax = vmax_rho or rho.max()
    pa = PolyCollection(qa, array=np.ravel(rho), cmap="viridis", norm=LogNorm(vmax * 1e-3, vmax), edgecolors="face",
                        linewidths=0.3)
    pb = PolyCollection(qb, array=np.ravel(Ti), cmap="inferno", norm=LogNorm(max(Ti.max() * 1e-3, 1e-3), Ti.max()),
                        edgecolors="face", linewidths=0.3)
    ax.add_collection(pa); ax.add_collection(pb)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_facecolor("black")
    ax.set_title(title, fontsize=9)
    ax.text(0.02, 0.97, "density", color="w", fontsize=7, transform=ax.transAxes, va="top")
    ax.text(0.98, 0.97, "Ti", color="w", fontsize=7, transform=ax.transAxes, va="top", ha="right")
    return pa, pb


def main(dirs, out=None):
    n = len(dirs)
    fig, axs = plt.subplots(n, 4, figsize=(15, 3.6 * n), squeeze=False)
    rows = []
    for k, d in enumerate(dirs):
        S = np.load(os.path.join(d, "snapshots2d.npz"))
        H = dict(np.load(os.path.join(d, "history.npz")))
        D = json.load(open(os.path.join(d, "summary.json")))["diagnostics"]
        name = os.path.basename(os.path.normpath(d))
        t = S["t"]
        tv = D["t_vmax_ns"] * 1e-9; tb = D["bang_time_ns"] * 1e-9
        iv = int(np.argmin(abs(t - tv))); ib = int(np.argmin(abs(t - tb)))
        dv = S["rho"][iv] > 0.05 * S["rho"][iv].max()
        rv = np.sqrt(S["zc"][iv] ** 2 + S["Rc"][iv] ** 2)[dv].max() * 1e4
        pa, pb = _map(axs[k, 0], S, iv, 1.5 * rv, f"{name}: t = {t[iv]*1e9:.2f} ns (peak velocity)")
        dense = S["rho"][ib] > 0.05 * S["rho"][ib].max()
        rr = np.sqrt(S["zc"][ib] ** 2 + S["Rc"][ib] ** 2)[dense].max() * 1e4
        pa, pb = _map(axs[k, 1], S, ib, 1.6 * rr, f"t = {t[ib]*1e9:.3f} ns (bang time)")
        fig.colorbar(pa, ax=axs[k, 0], fraction=0.046, pad=0.02, label="density [g/cc] (left half)")
        fig.colorbar(pb, ax=axs[k, 1], fraction=0.046, pad=0.02, label="Ti [keV] (right half)")
        # radial lineouts at bang: pole vs equator
        a = axs[k, 2]
        rc = np.sqrt(S["zc"][ib] ** 2 + S["Rc"][ib] ** 2) * 1e4
        Nj = rc.shape[1]
        for j, c, lab in [(0, "tab:blue", "pole"), (Nj - 1, "tab:red", "equator")]:
            a.semilogy(rc[:, j], S["rho"][ib][:, j], color=c, label=f"density, {lab}")
            a.semilogy(rc[:, j], S["Ti"][ib][:, j] / 1e3, color=c, ls="--", label=f"Ti [keV], {lab}")
        a.set_xlim(0, 1.6 * rr); a.set_ylim(1e-1, S["rho"][ib].max() * 2); a.legend(fontsize=6)
        a.set_xlabel("r [um]"); a.set_title("bang-time lineouts", fontsize=9)
        # fuel inner surface Legendre amplitudes
        a = axs[k, 3]
        th = H["t"] * 1e9
        for l, c in [(2, "C0"), (4, "C1"), (6, "C2")]:
            key = f"ice_in_P{l}"
            if key in H:
                # older runs projected r instead of r - <r>: remove the projection of a sphere
                tg = S["theta"]; Pl = np.polynomial.legendre.legval(np.cos(tg), np.eye(l + 1)[l])
                c0 = np.trapezoid(Pl * np.sin(tg), tg) / np.trapezoid(Pl * Pl * np.sin(tg), tg)
                a.plot(th, 100 * (H[key] - c0 * (abs(H[key][0] - c0) < 1e-9)), color=c, label=f"P{l}")
        a.axvline(D["bang_time_ns"], color="k", lw=0.6, ls=":")
        a.set_xlim(max(0, D["bang_time_ns"] - 3), D["bang_time_ns"] + 0.2)
        a.set_xlabel("t [ns]"); a.set_ylabel("ice inner surface a_l / R [%]"); a.legend(fontsize=7)
        Q = D["Q"]["Q_target"]
        a.set_title(f"yield {D['yield_MJ']:.2f} MJ, Q = {Q:.1f}", fontsize=9)
        rows.append((name, D))
    fig.tight_layout()
    out = out or os.path.join(os.path.dirname(os.path.normpath(dirs[0])), "krf2d_compare.png")
    fig.savefig(out, dpi=110)
    print(out)
    for name, D in rows:
        print(f"{name:20s} yield {D['yield_MJ']:7.3f} MJ  Q_target {D['Q']['Q_target']:6.2f}  "
              f"rhoR {D['rhoR_fuel_max']:.2f}  bang {D['bang_time_ns']:.3f} ns  Ti {D['Ti_hs_bang_keV']:.1f} keV")


if __name__ == "__main__":
    main(sys.argv[1:])
