"""SIFE - Simulation of Inertial Fusion Energy.

1D Lagrangian three-temperature radiation hydrodynamics for laser
(KrF) direct-drive inertial confinement fusion.
"""
from . import constants, materials
from .materials import Material, DT_ice, DT_gas, CH, CH_foam_DT, ideal
from .qvalue import q_values, q_history, format_q
from .simulation import Simulation, Options, Layer, Laser, build_mesh, tabulated_pulse

__version__ = "0.1.0"
