"""Versioned six-domain continuous-time neurobiological network.

The v1.0 numerical baseline is an affine continuous-time system:
    dx/dt = b + A x

A[target, source] is the directed influence of a source state on the
derivative of a target state. Coefficients are supplied explicitly; this
module does not assign biomarker mappings or biological coefficient values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm


DOMAIN_NAMES = ("P", "I", "M", "N", "Q", "Pr")
DOMAIN_INDEX = {name: index for index, name in enumerate(DOMAIN_NAMES)}

# Working sign convention for latent coordinates. These are semantic
# conventions, not biomarker definitions or claims about measured quantities.
DOMAIN_POLARITY = {
    "P": "higher = greater neurodegenerative progression burden",
    "I": "higher = greater inflammatory burden",
    "M": "higher = greater metabolic dysregulation",
    "N": "higher = greater preserved neuronal integrity",
    "Q": "higher = greater functional capacity",
    "Pr": "higher = greater proteostasis impairment",
}

Disease = Literal["alzheimer", "parkinson"]
ParameterStatus = Literal["hypothesis", "estimated", "validated"]


def _finite_vector(value: np.ndarray, *, name: str, size: int) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _finite_matrix(value: np.ndarray, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SixDomainNetwork:
    """Disease-specific six-state affine network.

    Parameters
    ----------
    disease:
        Separate parameterization for Alzheimer or Parkinson disease.
    time_unit:
        Unit used by timestamps and by the rates in coupling_matrix.
    drift:
        Constant state drift, in latent-coordinate units per time_unit.
    coupling_matrix:
        Directed 6x6 matrix in DOMAIN_NAMES order. Element [i, j] is the
        contribution of state j to the time derivative of state i.
    parameter_status:
        "hypothesis" is the safe default. "estimated" and "validated" are
        provenance labels and must only be set by a separately audited process.
    """

    disease: Disease
    time_unit: str
    drift: np.ndarray
    coupling_matrix: np.ndarray
    candidate_edges: tuple[tuple[str, str], ...]
    version: str = "1.0.0"
    parameter_status: ParameterStatus = "hypothesis"

    def __post_init__(self) -> None:
        if self.disease not in ("alzheimer", "parkinson"):
            raise ValueError("disease must be 'alzheimer' or 'parkinson'")
        if not isinstance(self.time_unit, str) or not self.time_unit.strip():
            raise ValueError("time_unit must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if self.parameter_status not in ("hypothesis", "estimated", "validated"):
            raise ValueError("invalid parameter_status")
        object.__setattr__(self, "drift", _finite_vector(self.drift, name="drift", size=6))
        matrix = _finite_matrix(self.coupling_matrix, name="coupling_matrix", shape=(6, 6))
        normalized_edges: list[tuple[str, str]] = []
        for edge in self.candidate_edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError("each candidate edge must be a (source, target) pair")
            source, target = edge
            if source not in DOMAIN_INDEX or target not in DOMAIN_INDEX:
                raise ValueError(f"candidate edge contains an unknown domain: {edge!r}")
            normalized_edges.append((source, target))
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError("candidate_edges must not contain duplicates")
        normalized_edges.sort(key=lambda edge: (DOMAIN_INDEX[edge[0]], DOMAIN_INDEX[edge[1]]))
        allowed_matrix_entries = {
            (DOMAIN_INDEX[target], DOMAIN_INDEX[source])
            for source, target in normalized_edges
        }
        nonzero_entries = set(map(tuple, np.argwhere(matrix != 0.0)))
        if not nonzero_entries.issubset(allowed_matrix_entries):
            raise ValueError(
                "coupling_matrix contains non-zero entries outside candidate_edges"
            )
        object.__setattr__(self, "coupling_matrix", matrix)
        object.__setattr__(self, "candidate_edges", tuple(normalized_edges))

    @property
    def state_names(self) -> tuple[str, ...]:
        return DOMAIN_NAMES

    @property
    def matrix_A(self) -> np.ndarray:
        """Return a writable copy of the directed coupling matrix."""
        return np.array(self.coupling_matrix, copy=True)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Stable parameter order for future calibration and audit records."""
        drift_names = tuple(f"drift:{name}" for name in DOMAIN_NAMES)
        edge_names = tuple(f"A:{source}->{target}" for source, target in self.candidate_edges)
        return drift_names + edge_names

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def as_parameter_vector(self) -> np.ndarray:
        """Return drift followed by candidate-edge coefficients in stable order."""
        edge_values = [
            self.coupling_matrix[DOMAIN_INDEX[target], DOMAIN_INDEX[source]]
            for source, target in self.candidate_edges
        ]
        return np.concatenate([self.drift, np.asarray(edge_values, dtype=float)])

    def with_parameter_vector(self, values: np.ndarray) -> "SixDomainNetwork":
        """Create a new immutable model with the same graph and new parameters."""
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.parameter_count,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"parameter vector must be finite with shape ({self.parameter_count},)")
        matrix = np.zeros((6, 6), dtype=float)
        for value, (source, target) in zip(vector[6:], self.candidate_edges):
            matrix[DOMAIN_INDEX[target], DOMAIN_INDEX[source]] = value
        return SixDomainNetwork(
            disease=self.disease,
            time_unit=self.time_unit,
            drift=vector[:6],
            coupling_matrix=matrix,
            candidate_edges=self.candidate_edges,
            version=self.version,
            parameter_status=self.parameter_status,
        )

    def rhs(self, state: np.ndarray) -> np.ndarray:
        """Evaluate dx/dt = drift + A @ state."""
        x = np.asarray(state, dtype=float)
        if x.shape != (6,) or not np.all(np.isfinite(x)):
            raise ValueError("state must be a finite vector with shape (6,)")
        return self.drift + self.coupling_matrix @ x


@dataclass(frozen=True)
class NetworkTrajectory:
    time: np.ndarray
    state: np.ndarray  # shape (n_time, 6), columns in DOMAIN_NAMES order


def _validate_time(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 1 or not np.all(np.isfinite(t)):
        raise ValueError("time must be a finite 1-D array with at least one point")
    if len(t) > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return t


def exact_affine_transition(
    network: SixDomainNetwork,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact (Phi, offset) for an interval of length dt.

    [x(t + dt)] = Phi @ x(t) + offset
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    augmented = np.zeros((7, 7), dtype=float)
    augmented[:6, :6] = network.coupling_matrix
    augmented[:6, 6] = network.drift
    exponential = expm(augmented * float(dt))
    return exponential[:6, :6], exponential[:6, 6]


def simulate_network(
    network: SixDomainNetwork,
    initial_state: np.ndarray,
    time: np.ndarray,
) -> NetworkTrajectory:
    """Propagate the affine network exactly at arbitrary increasing time points.

    Exact matrix-exponential propagation avoids time-step approximation for
    this affine v1.0 model and naturally supports irregular visit schedules.
    """
    t = _validate_time(time)
    x0 = np.asarray(initial_state, dtype=float)
    if x0.shape != (6,) or not np.all(np.isfinite(x0)):
        raise ValueError("initial_state must be a finite vector with shape (6,)")

    states = np.empty((len(t), 6), dtype=float)
    states[0] = x0
    for index, dt in enumerate(np.diff(t), start=1):
        phi, offset = exact_affine_transition(network, float(dt))
        states[index] = phi @ states[index - 1] + offset
    return NetworkTrajectory(time=t.copy(), state=states)
