"""Load FEDVR integrals exported by ``export_integrals.cpp``.

JSON schema is documented at the top of the C++ source.  The Python side keeps
all numerical data as ``numpy`` arrays and packs them into a small dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Nucleus:
    Z: float
    X: float


@dataclass
class Integrals:
    """One- and two-electron integrals on a 1D FEDVR DVR basis.

    Attributes
    ----------
    tag : str
        Free-form identifier set on the C++ side (used for filenames / logs).
    L, n_elements, n_order : geometry of the FEDVR grid (see ``FEDVRGrid``).
    N_DVR : int
        Number of independent DVR points (= ``n_elements * n_order - 1``).
        The qubit register has ``2 * N_DVR`` qubits (spin-up and spin-down per
        spatial orbital).
    N_e : int
        Number of electrons (used to pick the particle-number sector when
        diagonalising; does not affect Hamiltonian construction).
    nuclei : list[Nucleus]
        Nuclear charges and 1D positions used in the soft-Coulomb attraction.
    E_nn : float
        Nuclear-nuclear repulsion sum (soft-Coulomb, ``a = 1``).
    x, w : np.ndarray, shape (N_DVR,)
        DVR node coordinates and quadrature weights.
    h_pq : np.ndarray, shape (N_DVR, N_DVR)
        One-electron matrix (kinetic + nuclear attraction) in the DVR basis.
    V_pq : np.ndarray, shape (N_DVR, N_DVR)
        Two-electron repulsion *kernel* with DVR contraction
        ``V_{pqrs} = δ_{pr} δ_{qs} V_{pq}``,  V_pq = 1/sqrt((x_p - x_q)^2 + 1).
    """

    tag: str
    L: float
    n_elements: int
    n_order: int
    N_DVR: int
    N_e: int
    nuclei: list[Nucleus]
    E_nn: float
    x: np.ndarray
    w: np.ndarray
    h_pq: np.ndarray
    V_pq: np.ndarray

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the spin-orbital register (= 2 * N_DVR)."""
        return 2 * self.N_DVR

    def summary(self) -> str:
        return (
            f"Integrals(tag={self.tag!r}, L={self.L}, "
            f"n_elements={self.n_elements}, n_order={self.n_order}, "
            f"N_DVR={self.N_DVR}, n_qubits={self.n_qubits}, "
            f"N_e={self.N_e}, nuclei={self.nuclei}, E_nn={self.E_nn:.6f})"
        )


def load(path: str | Path) -> Integrals:
    """Load a JSON file produced by ``export_integrals.cpp``."""
    raw = json.loads(Path(path).read_text())
    model: dict[str, Any] = raw["model"]
    nuclei = [Nucleus(Z=float(n["Z"]), X=float(n["X"])) for n in model["nuclei"]]

    x = np.asarray(raw["x"], dtype=float)
    w = np.asarray(raw["w"], dtype=float)
    h_pq = np.asarray(raw["h_pq"], dtype=float)
    V_pq = np.asarray(raw["V_pq"], dtype=float)

    N = int(model["N_DVR"])
    if x.shape != (N,) or w.shape != (N,):
        raise ValueError(f"x/w shape mismatch: {x.shape}/{w.shape} vs N={N}")
    if h_pq.shape != (N, N) or V_pq.shape != (N, N):
        raise ValueError(f"h_pq/V_pq shape mismatch: {h_pq.shape}/{V_pq.shape} vs N={N}")

    # Sanity: h_pq should be symmetric, V_pq should be symmetric and have V_pp = 1
    sym_err = float(np.max(np.abs(h_pq - h_pq.T)))
    if sym_err > 1e-10:
        raise ValueError(f"h_pq not symmetric (max asym = {sym_err:.2e})")
    if not np.allclose(np.diag(V_pq), 1.0, atol=1e-12):
        raise ValueError("V_pq diagonal should be 1 (V_pp = 1/sqrt(0+1) = 1)")

    return Integrals(
        tag=str(model["tag"]),
        L=float(model["L"]),
        n_elements=int(model["n_elements"]),
        n_order=int(model["n_order"]),
        N_DVR=N,
        N_e=int(model["N_e"]),
        nuclei=nuclei,
        E_nn=float(model["E_nn"]),
        x=x,
        w=w,
        h_pq=h_pq,
        V_pq=V_pq,
    )
