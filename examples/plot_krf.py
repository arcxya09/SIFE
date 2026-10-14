"""Plots for a KrF target run: python plot_krf.py <output dir>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3})

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "output", "krf")
h = dict(np.load(os.path.join(d, "history.npz")))
s = dict(np.load(os.path.join(d, "snapshots.npz")))
summ = json.load(open(os.path.join(d, "summary.json")))
D = summ["diagnostics"]
t = h["t"] * 1e9
mat = s["mat"]
ts = s["t"] * 1e9
tb = D["bang_time_ns"]

# ---------------------------------------------------------------- overview
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
a = ax[0, 0]
a.plot(t, h["P_laser"] / 1e19, "k-", lw=1, label="incident")
a.plot(t, h["P_abs"] / 1e19, "C3-", lw=1, label="absorbed")
a.set_xlabel("t [ns]"); a.set_ylabel("power [TW]"); a.legend()
a.set_title(f"KrF pulse: {D['E_laser_MJ']:.3f} MJ, absorption {100*D['absorption']:.1f}%")

a = ax[0, 1]
r = s["r"] * 1e4
nn = r.shape[1]
iface = [np.where(mat == k)[0][0] for k in np.unique(mat)] + [len(mat)]
for j in range(0, nn, 4):
    a.plot(ts, r[:, j], color="0.75", lw=0.4)
for k, j in enumerate(iface):
    a.plot(ts, r[:, j], color=f"C{k}", lw=1.4)
if "R_crit" in h:
    a.plot(t, h["R_crit"] * 1e4, "m--", lw=1, label="critical surface")
a.set_ylim(0, 1.25 * r[0, -1]); a.set_xlabel("t [ns]"); a.set_ylabel("r [µm]")
a.set_title("Lagrangian trajectories (coloured: material interfaces)"); a.legend(loc="upper right")

a = ax[0, 2]
a.plot(t, h["v_imp"] / 1e7, "C0-", label=r"shell velocity [10$^7$ cm/s]")
a.plot(t, h["rhoR_fuel"], "C2-", label=r"fuel $\rho R$ [g/cm$^2$]")
a.set_xlabel("t [ns]"); a.legend()
a.set_title(f"v_imp,max = {D['v_imp_max_cm_s']/1e7:.2f}e7 cm/s, ρR_max = {D['rhoR_fuel_max']:.2f}")

a = ax[1, 0]
a.semilogy(t, np.maximum(h["Ti_hs"], 1) / 1e3, "C3-", label="hot-spot <Ti> [keV]")
a.semilogy(t, h["rho_max"], "C0-", label=r"$\rho_{max}$ [g/cc]")
a.set_xlabel("t [ns]"); a.legend(); a.set_xlim(max(0, tb - 2.0), t[-1])
a.set_title(f"stagnation: Ti(bang) = {D['Ti_hs_bang_keV']:.1f} keV, ρmax = {D['rho_max']:.0f} g/cc")

a = ax[1, 1]
a.semilogy(t, np.maximum(h["E_fus"], 1) / 1e13, "C1-", label="fusion yield [MJ]")
a.semilogy(t, np.maximum(h["reac_rate"], 1) / 1e30, "C4-", label=r"reaction rate [10$^{30}$/s]")
a.set_xlim(max(0, tb - 1.0), min(t[-1], tb + 1.0)); a.set_xlabel("t [ns]"); a.legend()
a.set_title(f"yield {D['yield_MJ']:.2f} MJ, gain {D['gain']:.1f}, burn-up {100*D['burnup']:.1f}%")

a = ax[1, 2]
E = [h["KE_in"], h["KE"], h["IE"] - h["IE"][0], h["E_rad_lost"], h["E_abs"]]
lab = ["implosion KE (u<0)", "total kinetic", "internal (Δ)", "radiation lost", "laser absorbed"]
for e_, l in zip(E, lab):
    a.plot(t, e_ / 1e10, label=l)
a.set_xlabel("t [ns]"); a.set_ylabel("kJ"); a.legend(); a.set_xlim(0, tb - 0.03)
a.set_ylim(0, 1.15 * h["E_abs"].max() / 1e10)
a.set_title(f"energy budget (shell KE {D['KE_shell_max_kJ']:.0f} kJ, η_hydro {100*D['hydro_eff']:.1f}%)")
fig.tight_layout(); fig.savefig(os.path.join(d, "overview.png"))

# ---------------------------------------------------------------- profiles
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
times = [1.0, 4.0, D["t_vmax_ns"], tb]
labels = ["foot (1 ns)", "4 ns", "peak velocity", "bang time"]
for k, (tt, lab) in enumerate(zip(times, labels)):
    i = int(np.argmin(abs(ts - tt)))
    rc = 0.5 * (s["r"][i, 1:] + s["r"][i, :-1]) * 1e4
    ax[0, 0].semilogy(rc, s["rho"][i], label=f"{lab}, t={ts[i]:.2f} ns")
    ax[0, 1].semilogy(rc, s["Te"][i] / 1e3, f"C{k}-")
    ax[0, 1].semilogy(rc, s["Ti"][i] / 1e3, f"C{k}:")
    ax[1, 0].semilogy(rc, np.maximum(s["P"][i] / 1e12, 1e-6), f"C{k}-")
    ax[1, 1].plot(s["r"][i] * 1e4, s["u"][i] / 1e7, f"C{k}-")
ax[0, 0].set_ylabel(r"$\rho$ [g/cc]"); ax[0, 0].legend(fontsize=7)
ax[0, 1].set_ylabel("Te (solid), Ti (dotted) [keV]")
ax[1, 0].set_ylabel("P [Mbar]"); ax[1, 1].set_ylabel(r"u [10$^7$ cm/s]")
for a in ax.flat:
    a.set_xlabel("r [µm]"); a.set_xlim(0, 1.6 * s["r"][0, -1] * 1e4)
fig.suptitle("Radial profiles")
fig.tight_layout(); fig.savefig(os.path.join(d, "profiles.png"))

# stagnation zoom
i = int(np.argmin(abs(ts - tb)))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
rc = 0.5 * (s["r"][i, 1:] + s["r"][i, :-1]) * 1e4
ax[0].plot(rc, s["rho"][i], "C0-"); ax[0].set_ylabel(r"$\rho$ [g/cc]", color="C0")
b = ax[0].twinx(); b.plot(rc, s["Ti"][i] / 1e3, "C3-"); b.set_ylabel("Ti [keV]", color="C3")
ax[0].set_xlim(0, 3 * max(D["R_hs_bang_um"], 30)); ax[0].set_xlabel("r [µm]")
ax[0].set_title(f"bang time {tb:.3f} ns: hot spot R = {D['R_hs_bang_um']:.0f} µm")
ax[1].plot(rc, s["burn"][i], "C1-"); ax[1].set_xlabel("r [µm]"); ax[1].set_ylabel("local burn-up fraction")
j = len(ts) - 1
rcj = 0.5 * (s["r"][j, 1:] + s["r"][j, :-1]) * 1e4
ax[1].plot(rcj, s["burn"][j], "C1--", label=f"final t={ts[j]:.2f} ns"); ax[1].legend()
ax[1].set_xlim(0, 3 * max(D["R_hs_bang_um"], 30) * 3)
fig.tight_layout(); fig.savefig(os.path.join(d, "stagnation.png"))
print("plots written to", d)
