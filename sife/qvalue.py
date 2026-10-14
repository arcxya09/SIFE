"""Energy gain (Q) bookkeeping for laser-driven ICF.

Definitions (all energies per shot):

  Q_target  = E_fus / E_laser          target (scientific) gain, the usual "G" of ICF
  Q_abs     = E_fus / E_abs            gain on the laser energy actually absorbed
  Q_kin     = E_fus / KE_shell_max     gain on the peak shell kinetic energy
  Q_fuel    = E_fus / E_fuel           gain on the energy delivered to the DT fuel (peak fuel
                                       kinetic energy; the NIF 'fuel gain', > 1 since 2014)
  eta*G     = eta_driver * Q_target    "wall-plug" gain; inertial fusion energy needs eta*G >~ 10
  Q_eng     = eta_th * M * eta_driver * Q_target
                                       engineering gain: electricity produced / electricity
                                       consumed by the driver (M = blanket energy multiplication)
  f_recirc  = 1 / Q_eng                recirculating power fraction

Defaults: KrF wall-plug efficiency 7 % (Electra/NRL goal; ArF ~ 10 %),
thermal-to-electric conversion 40 %, blanket multiplication 1.13 (Li/Pb-Li).
"""
import numpy as np

KRF_ETA_DRIVER = 0.07
ETA_THERMAL = 0.40
BLANKET_MULT = 1.13


def q_values(E_fus, E_laser, E_abs=None, KE_shell=None, E_fuel=None,
             eta_driver=KRF_ETA_DRIVER, eta_thermal=ETA_THERMAL, blanket_mult=BLANKET_MULT):
    """Return a dict of gain figures.  Energies in any common unit."""
    tiny = 1e-300
    d = {}
    d["Q_target"] = E_fus / max(E_laser, tiny)
    if E_abs is not None:
        d["Q_abs"] = E_fus / max(E_abs, tiny)
    if KE_shell is not None:
        d["Q_kin"] = E_fus / max(KE_shell, tiny)
    if E_fuel is not None:
        d["Q_fuel"] = E_fus / max(E_fuel, tiny)
    d["eta_driver"] = eta_driver
    d["eta_G"] = eta_driver * d["Q_target"]
    d["Q_eng"] = eta_thermal * blanket_mult * eta_driver * d["Q_target"]
    d["f_recirc"] = 1.0 / d["Q_eng"] if d["Q_eng"] > 0 else float("inf")
    # net electricity per shot (MJ) if energies are given in erg
    d["E_electric_net_per_shot_MJ"] = (eta_thermal * blanket_mult * E_fus - E_laser / eta_driver) / 1e13
    d["eta_thermal"] = eta_thermal
    d["blanket_mult"] = blanket_mult
    return d


def q_history(h):
    """Time-resolved Q_target(t) = E_fus(t)/E_laser(t) and Q_abs(t) from a history dict."""
    EL = np.maximum(np.asarray(h["E_laser"]), 1e-300)
    EA = np.maximum(np.asarray(h["E_abs"]), 1e-300)
    Ef = np.asarray(h["E_fus"])
    return Ef / EL, Ef / EA


def format_q(d):
    lines = [f"Q_target (E_fus/E_laser) = {d['Q_target']:.3g}"]
    if "Q_abs" in d:
        lines.append(f"Q_abs    (E_fus/E_abs)   = {d['Q_abs']:.3g}")
    if "Q_kin" in d:
        lines.append(f"Q_kin    (E_fus/KE_shell)= {d['Q_kin']:.3g}")
    if "Q_fuel" in d:
        lines.append(f"Q_fuel   (E_fus/E_fuel)  = {d['Q_fuel']:.3g}")
    lines.append(f"eta*G    (eta={d['eta_driver']:.2f})     = {d['eta_G']:.3g}")
    rec = f"recirculating {100*d['f_recirc']:.0f} %" if d["Q_eng"] > 1e-6 else "no net output"
    lines.append(f"Q_eng                    = {d['Q_eng']:.3g}  ({rec})")
    return "\n".join(lines)
