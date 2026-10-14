"""Animations of SIFE runs (MP4 via ffmpeg, or GIF via Pillow).

    python -m sife.animate output/krf                 # 1D run  -> output/krf/implosion.mp4 (+ .gif)
    python -m sife.animate output/krf2d               # 2D run  -> output/krf2d/implosion2d.mp4

1D frames: density, temperatures, pressure and velocity profiles, the
Lagrangian r-t diagram and the laser / fusion power with the running gain Q(t).
2D frames: density and ion temperature maps of the (z, r) half plane mirrored
to the full meridional plane, plus the running Q(t).
"""
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm

from .qvalue import q_history

MAT_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange"]


def _writer(path, fps):
    if path.endswith(".mp4") and shutil.which("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                                      extra_args=["-pix_fmt", "yuv420p"])
    return animation.PillowWriter(fps=fps)


def _smooth_limits(x, n=7):
    lx = np.log(np.maximum(x, 1e-30))
    k = np.ones(n) / n
    pad = np.pad(lx, (n // 2, n // 2), mode="edge")
    return np.exp(np.convolve(pad, k, mode="valid"))


def _gif_from_mp4(mp4, gif, width=640, fps=12):
    if not shutil.which("ffmpeg"):
        return False
    pal = gif + ".palette.png"
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    os.system(f'ffmpeg -loglevel error -y -i "{mp4}" -vf "{vf},palettegen" "{pal}"')
    os.system(f'ffmpeg -loglevel error -y -i "{mp4}" -i "{pal}" -lavfi "{vf}[x];[x][1:v]paletteuse" "{gif}"')
    if os.path.exists(pal):
        os.remove(pal)
    return os.path.exists(gif)


# ------------------------------------------------------------------------- 1D
def animate_1d(run_dir, out=None, fps=20, gif=True, max_frames=600):
    S = np.load(os.path.join(run_dir, "snapshots.npz"))
    H = dict(np.load(os.path.join(run_dir, "history.npz")))
    t = S["t"]; r = S["r"] * 1e4  # um
    mat = S["mat"]
    nf = len(t)
    idx = np.arange(nf) if nf <= max_frames else np.linspace(0, nf - 1, max_frames).astype(int)
    out = out or os.path.join(run_dir, "implosion.mp4")
    Qt, Qa = q_history(H)
    th = H["t"] * 1e9
    # x-range: follow the dense shell (3x the radius of the outermost dense zone)
    xmax = np.empty(nf)
    for i in range(nf):
        rho = S["rho"][i]
        k = np.where(rho > 0.05 * rho.max())[0][-1]
        xmax[i] = min(r[i, -1] * 1.02, max(2.0 * r[i, k + 1], 60.0))
    xmax = _smooth_limits(xmax)

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.3)
    axd, axT, axP = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axu, axrt, axpw = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    # static r-t diagram
    nn = r.shape[1]
    for j in range(0, nn, max(1, nn // 60)):
        axrt.plot(t * 1e9, r[:, j], lw=0.4, color="0.4")
    if "R_crit" in H:
        axrt.plot(th, H["R_crit"] * 1e4, color="tab:orange", lw=1.2, label="critical surface")
    jf = np.where(mat == 1)[0]
    if len(jf):
        axrt.plot(t * 1e9, r[:, jf[0]], color="tab:blue", lw=1.2, label="fuel inner edge")
        axrt.plot(t * 1e9, r[:, jf[-1] + 1], color="tab:green", lw=1.2, label="fuel/ablator")
    axrt.set_xlabel("t [ns]"); axrt.set_ylabel("r [um]"); axrt.set_ylim(0, r[0, -1] * 1.05)
    axrt.set_xlim(t[0] * 1e9, t[-1] * 1e9); axrt.legend(fontsize=7, loc="upper right")
    axrt.set_title("Lagrangian r-t diagram", fontsize=10)
    cur_rt = axrt.axvline(0, color="k", lw=1)
    # static power / Q panel
    axpw.semilogy(th, np.maximum(H["P_laser"], 1e-30) * 1e-19, color="tab:purple", label="laser [TW]")
    Pf = np.gradient(H["E_fus"], H["t"]) * 1e-19
    axpw.semilogy(th, np.maximum(Pf, 1e-30), color="tab:red", label="fusion [TW]")
    axpw.set_ylim(1e-2, max(1e3, Pf.max() * 3)); axpw.set_xlabel("t [ns]"); axpw.set_ylabel("power [TW]")
    axq = axpw.twinx()
    axq.semilogy(th, np.maximum(Qt, 1e-8), color="k", ls="--", label="Q = E_fus/E_laser")
    axq.set_ylabel("Q(t)"); axq.set_ylim(1e-6, max(1.0, Qt.max() * 3))
    axq.axhline(1.0, color="0.5", lw=0.6, ls=":")
    l1, b1 = axpw.get_legend_handles_labels(); l2, b2 = axq.get_legend_handles_labels()
    axpw.legend(l1 + l2, b1 + b2, fontsize=7, loc="upper left")
    axpw.set_xlim(th[0], th[-1]); axpw.set_title("laser, fusion power and gain", fontsize=10)
    cur_pw = axpw.axvline(0, color="k", lw=1)

    rc = 0.5 * (r[:, 1:] + r[:, :-1])
    rho_all = S["rho"]; P_all = S["P"] / 1e15  # Gbar
    T_all = np.maximum(np.stack([S["Te"], S["Ti"], S["Tr"]]), 1e-3) / 1e3  # keV
    u_all = S["u"] / 1e7
    lines = {}
    segs = [np.where(mat == m)[0] for m in np.unique(mat)]
    lines["rho"] = [axd.semilogy([], [], color=MAT_COLORS[m % 5], lw=1.5)[0] for m in np.unique(mat)]
    axd.set_ylabel("density [g/cc]"); axd.set_ylim(1e-4, max(10.0, rho_all.max() * 2))
    lines["T"] = [axT.semilogy([], [], color=c, lw=1.4, label=lab)[0]
                  for c, lab in zip(["tab:orange", "tab:red", "tab:gray"], ["Te", "Ti", "Tr"])]
    axT.set_ylabel("T [keV]"); axT.set_ylim(1e-4, max(10.0, T_all.max() * 1.5)); axT.legend(fontsize=7)
    lines["P"] = axP.semilogy([], [], color="k", lw=1.4)[0]
    axP.set_ylabel("pressure [Gbar]"); axP.set_ylim(1e-6, max(1.0, P_all.max() * 2))
    lines["u"] = axu.plot([], [], color="tab:blue", lw=1.4)[0]
    axu.axhline(0, color="0.6", lw=0.5)
    axu.set_ylabel("velocity [10^7 cm/s]")
    axu.set_ylim(u_all.min() * 1.15, min(max(1.0, u_all.max() * 1.1), 2.0 * abs(u_all.min())))
    for a in (axd, axT, axP, axu):
        a.set_xlabel("r [um]")
    names = ["gas", "DT ice", "ablator", "skin"]
    axd.legend(lines["rho"], [names[m] if m < 4 else str(m) for m in np.unique(mat)], fontsize=7)
    title = fig.suptitle("")
    Ef = np.interp(t, H["t"], H["E_fus"]); El = np.interp(t, H["t"], H["E_laser"])

    def draw(i):
        x = rc[i]
        for ln, sg in zip(lines["rho"], segs):
            ln.set_data(x[sg], rho_all[i, sg])
        for ln, T in zip(lines["T"], T_all[:, i]):
            ln.set_data(x, T)
        lines["P"].set_data(x, P_all[i])
        lines["u"].set_data(r[i], u_all[i])
        for a in (axd, axT, axP, axu):
            a.set_xlim(0, xmax[i])
        cur_rt.set_xdata([t[i] * 1e9] * 2); cur_pw.set_xdata([t[i] * 1e9] * 2)
        Q = Ef[i] / max(El[i], 1e-30)
        title.set_text(f"SIFE  KrF direct drive    t = {t[i]*1e9:6.3f} ns    "
                       f"E_laser = {El[i]/1e13:5.3f} MJ    yield = {Ef[i]/1e13:7.3f} MJ    Q = {Q:7.3g}")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=idx, blit=False)
    ani.save(out, writer=_writer(out, fps), dpi=100)
    plt.close(fig)
    res = [out]
    if gif and out.endswith(".mp4"):
        g = out[:-4] + ".gif"
        if _gif_from_mp4(out, g, width=780, fps=min(fps, 15)):
            res.append(g)
    return res


# ------------------------------------------------------------------------- 2D
def _quads(z, R):
    """Polygons (Ni*Nj, 4, 2) of a logically rectangular node mesh (z, R) [Ni+1, Nj+1]."""
    p = np.stack([z, R], -1)
    q = np.stack([p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]], 2)
    return q.reshape(-1, 4, 2)


def animate_2d(run_dir, out=None, fps=15, gif=True, max_frames=500, fields=("rho", "Ti")):
    S = np.load(os.path.join(run_dir, "snapshots2d.npz"))
    H = dict(np.load(os.path.join(run_dir, "history.npz")))
    t = S["t"]; nf = len(t)
    idx = np.arange(nf) if nf <= max_frames else np.linspace(0, nf - 1, max_frames).astype(int)
    out = out or os.path.join(run_dir, "implosion2d.mp4")
    Qt, _ = q_history(H)
    th = H["t"] * 1e9
    z = S["z"] * 1e4; R = S["R"] * 1e4
    rad = np.sqrt(z ** 2 + R ** 2)
    xmax = np.empty(nf)
    for i in range(nf):
        rho = S["rho"][i]
        dense = rho > 0.05 * rho.max()
        rr = np.sqrt(S["zc"][i] ** 2 + S["Rc"][i] ** 2)[dense] * 1e4
        xmax[i] = min(rad[i].max() * 1.02, max(2.2 * rr.max(), 60.0))
    xmax = _smooth_limits(xmax)

    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.35, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0]); ax.set_aspect("equal")
    axr = fig.add_subplot(gs[0, 1]); axpw = fig.add_subplot(gs[1, 1])
    unit = {"rho": "density [g/cc]", "Ti": "Ti [keV]", "Te": "Te [keV]", "P": "P [Gbar]"}
    scale = {"rho": 1.0, "Ti": 1e-3, "Te": 1e-3, "P": 1e-15}
    fa, fb = fields
    va = S[fa] * scale[fa]; vb = S[fb] * scale[fb]
    norm_a = LogNorm(vmin=max(va.max() * 1e-5, 1e-4), vmax=va.max())
    norm_b = LogNorm(vmin=max(vb.max() * 1e-4, 1e-3), vmax=vb.max())
    pa = PolyCollection([], cmap="viridis", norm=norm_a, edgecolors="none", rasterized=True)
    pb = PolyCollection([], cmap="inferno", norm=norm_b, edgecolors="none", rasterized=True)
    mesh = LineCollection([], colors=[(1, 1, 1, 0.25)], linewidths=0.3)
    ax.add_collection(pa); ax.add_collection(pb); ax.add_collection(mesh)
    fig.colorbar(pa, ax=ax, location="left", shrink=0.7, pad=0.2, label=unit[fa])
    fig.colorbar(pb, ax=ax, location="right", shrink=0.7, pad=0.02, label=unit[fb])
    ax.set_xlabel("x [um]  (left: %s, right: %s)" % (fa, fb)); ax.set_ylabel("z [um]  (axis)")
    ax.set_facecolor("black")
    # radial lineouts along pole / equator
    Nj = S["rho"].shape[2]
    lp = axr.semilogy([], [], color="tab:blue", label="pole (theta=0)")[0]
    le = axr.semilogy([], [], color="tab:red", label="equator")[0]
    axr.set_ylabel("density [g/cc]"); axr.set_xlabel("radius [um]"); axr.legend(fontsize=7)
    axr.set_ylim(1e-4, S["rho"].max() * 2)
    axpw.semilogy(th, np.maximum(H["P_laser"], 1e-30) * 1e-19, color="tab:purple", label="laser [TW]")
    Pf = np.gradient(H["E_fus"], H["t"]) * 1e-19
    axpw.semilogy(th, np.maximum(Pf, 1e-30), color="tab:red", label="fusion [TW]")
    axpw.set_ylim(1e-2, max(1e3, Pf.max() * 3)); axpw.set_xlabel("t [ns]")
    axq = axpw.twinx()
    axq.semilogy(th, np.maximum(Qt, 1e-8), "k--", label="Q(t)")
    axq.set_ylim(1e-6, max(1.0, Qt.max() * 3)); axq.set_ylabel("Q = E_fus/E_laser")
    l1, b1 = axpw.get_legend_handles_labels(); l2, b2 = axq.get_legend_handles_labels()
    axpw.legend(l1 + l2, b1 + b2, fontsize=7, loc="upper left")
    cur = axpw.axvline(0, color="k", lw=1)
    title = fig.suptitle("")
    Ef = np.interp(t, H["t"], H["E_fus"]); El = np.interp(t, H["t"], H["E_laser"])

    half = bool(S["half"]) if "half" in S else False
    mat = S["mat"][:, 0] if "mat" in S else np.zeros(z.shape[1] - 1)
    ale = "X" in S                          # ALE run: material interfaces from the mass fractions
    if ale:
        rings = sorted(set(range(0, z.shape[1], 10)) | {z.shape[1] - 1})
        iface = LineCollection([], colors=[(1, 1, 1, 0.9)], linewidths=0.8)
        ax.add_collection(iface)
    else:
        rings = sorted(set(range(0, z.shape[1], 10)) | set(np.nonzero(np.diff(mat))[0] + 1) | {z.shape[1] - 1})

    def interfaces(i):
        X = S["X"][i]; zc = S["zc"][i] * 1e4; Rc = S["Rc"][i] * 1e4
        segs = []
        for k in range(X.shape[0] - 1):
            f = X[:k + 1].sum(0)            # fraction of materials 0..k
            pts = []
            for j in range(f.shape[1]):
                c = np.nonzero((f[:-1, j] >= 0.5) & (f[1:, j] < 0.5))[0]
                if len(c) == 0:
                    continue
                a = c[-1]; w = (f[a, j] - 0.5) / max(f[a, j] - f[a + 1, j], 1e-30)
                pts.append([Rc[a, j] + w * (Rc[a + 1, j] - Rc[a, j]), zc[a, j] + w * (zc[a + 1, j] - zc[a, j])])
            if len(pts) > 1:
                segs.append(np.array(pts))
        return segs

    def draw(i):
        Z, X = z[i], R[i]
        qa = _quads(Z, -X)[:, :, ::-1]      # left half: x = -R
        qb = _quads(Z, X)[:, :, ::-1]       # right half: x = +R
        A = va[i].ravel(); B = vb[i].ravel()
        if half:                            # mirror the hemisphere through the equator
            flip = np.array([1.0, -1.0])
            qa = np.concatenate([qa, qa * flip]); qb = np.concatenate([qb, qb * flip])
            A = np.concatenate([A, A]); B = np.concatenate([B, B])
        pa.set_verts(qa); pa.set_array(A)
        pb.set_verts(qb); pb.set_array(B)
        # sparse mesh on the right half: every spoke, every 10th ring and the material interfaces
        segs = [np.c_[X[i_], Z[i_]] for i_ in rings] + [np.c_[X[:, j], Z[:, j]] for j in range(X.shape[1])]
        if half:
            segs += [sg * np.array([1.0, -1.0]) for sg in segs]
        mesh.set_segments(segs)
        if ale:
            sg = interfaces(i)
            if half:
                sg += [p * np.array([1.0, -1.0]) for p in sg]
            iface.set_segments(sg)
        L = xmax[i]
        ax.set_xlim(-L, L); ax.set_ylim(-L, L)
        rr = np.sqrt(S["zc"][i] ** 2 + S["Rc"][i] ** 2) * 1e4
        je = Nj - 1 if half else Nj // 2
        lp.set_data(rr[:, 0], S["rho"][i][:, 0]); le.set_data(rr[:, je], S["rho"][i][:, je])
        axr.set_xlim(0, L)
        cur.set_xdata([t[i] * 1e9] * 2)
        Q = Ef[i] / max(El[i], 1e-30)
        title.set_text(f"SIFE-2D  KrF direct drive    t = {t[i]*1e9:6.3f} ns    "
                       f"yield = {Ef[i]/1e13:7.3f} MJ    Q = {Q:7.3g}")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=idx, blit=False)
    ani.save(out, writer=_writer(out, fps), dpi=100)
    plt.close(fig)
    res = [out]
    if gif and out.endswith(".mp4"):
        g = out[:-4] + ".gif"
        if _gif_from_mp4(out, g, width=720, fps=min(fps, 12)):
            res.append(g)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Make an animation of a SIFE run directory")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output .mp4 or .gif")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.run_dir, "snapshots2d.npz")):
        res = animate_2d(a.run_dir, a.out, fps=a.fps, gif=not a.no_gif)
    else:
        res = animate_1d(a.run_dir, a.out, fps=a.fps, gif=not a.no_gif)
    print("\n".join(res))


if __name__ == "__main__":
    main()
