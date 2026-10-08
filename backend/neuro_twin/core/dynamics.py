"""Numerical and sensitivity engine for the P-I-N-Q dynamics.

The canonical model is affine-linear in the state:
    dx/dt = c(theta) + A(theta)x

This admits stable propagation and an exact sensitivity ODE without finite-
difference derivatives of the RHS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from neuro_twin.core.parameters import PARAMETER_NAMES, PINQParameters


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray
    state: np.ndarray  # shape (n_time, 4)


@dataclass(frozen=True)
class SensitivityTrajectory:
    time: np.ndarray
    state: np.ndarray  # (n_time, 4)
    sensitivity: np.ndarray  # (n_time, 4, 11)


def rhs(x: np.ndarray, theta: PINQParameters) -> np.ndarray:
    P, I, N, Q = x
    return np.array([
        theta.aP + theta.bPI * I - theta.dP * P,
        theta.bIP * P - theta.dI * I,
        theta.bNP * P + theta.bNI * I - theta.dN * N,
        theta.bQN * N + theta.bQI * I - theta.dQ * Q,
    ], dtype=float)


def jacobian_x(theta: PINQParameters) -> np.ndarray:
    # Exact Jacobian ∂f/∂x.
    return theta.matrix_A()


def jacobian_theta(x: np.ndarray, theta: PINQParameters) -> np.ndarray:
    """Exact ∂f/∂theta with columns in PARAMETER_NAMES order."""
    P, I, N, Q = x
    out = np.zeros((4, len(PARAMETER_NAMES)), dtype=float)
    out[0, 0] = 1.0       # aP
    out[0, 1] = I         # bPI
    out[0, 2] = -P        # dP
    out[1, 3] = P         # bIP
    out[1, 4] = -I        # dI
    out[2, 5] = P         # bNP
    out[2, 6] = I         # bNI
    out[2, 7] = -N        # dN
    out[3, 8] = N         # bQN
    out[3, 9] = I         # bQI
    out[3, 10] = -Q       # dQ
    return out


def integrate(x0: np.ndarray, time: np.ndarray, theta: PINQParameters, *, rtol: float = 1e-8, atol: float = 1e-10) -> Trajectory:
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing 1-D array with at least two points")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (4,):
        raise ValueError("x0 must have shape (4,)")
    sol = solve_ivp(
        fun=lambda t, x: rhs(x, theta),
        t_span=(float(time[0]), float(time[-1])),
        y0=x0,
        t_eval=time,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return Trajectory(time=sol.t, state=sol.y.T)


def integrate_with_sensitivities(
    x0: np.ndarray,
    time: np.ndarray,
    theta: PINQParameters,
    *,
    initial_sensitivity: np.ndarray | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> SensitivityTrajectory:
    """Integrate state and exact forward sensitivities S=∂x/∂θ."""
    time = np.asarray(time, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    S0 = np.zeros((4, len(PARAMETER_NAMES))) if initial_sensitivity is None else np.asarray(initial_sensitivity, dtype=float)
    if x0.shape != (4,) or S0.shape != (4, len(PARAMETER_NAMES)):
        raise ValueError("invalid initial state/sensitivity shape")

    y0 = np.concatenate([x0, S0.ravel()])

    def augmented_rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:4]
        S = y[4:].reshape(4, len(PARAMETER_NAMES))
        dx = rhs(x, theta)
        dS = jacobian_x(theta) @ S + jacobian_theta(x, theta)
        return np.concatenate([dx, dS.ravel()])

    sol = solve_ivp(
        augmented_rhs,
        (float(time[0]), float(time[-1])),
        y0,
        t_eval=time,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    states = sol.y[:4].T
    sens = sol.y[4:].T.reshape(-1, 4, len(PARAMETER_NAMES))
    return SensitivityTrajectory(time=sol.t, state=states, sensitivity=sens)


def integrate_with_state_transition(
    x0: np.ndarray,
    time: np.ndarray,
    theta: PINQParameters,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate state plus Phi=∂x(t)/∂x0 exactly via dPhi/dt=A Phi."""
    time = np.asarray(time, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (4,):
        raise ValueError("x0 must have shape (4,)")
    y0 = np.concatenate([x0, np.eye(4).ravel()])
    A = jacobian_x(theta)

    def augmented_rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:4]
        Phi = y[4:].reshape(4, 4)
        dx = rhs(x, theta)
        dPhi = A @ Phi
        return np.concatenate([dx, dPhi.ravel()])

    sol = solve_ivp(
        augmented_rhs,
        (float(time[0]), float(time[-1])),
        y0,
        t_eval=time,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:4].T, sol.y[4:].T.reshape(-1, 4, 4)


def propagate_affine_expm(x0: np.ndarray, time: np.ndarray, theta: PINQParameters) -> Trajectory:
    """Independent closed-form propagator for dx/dt = A x + c.

    The augmented matrix
        M = [[A, c], [0, 0]]
    gives [x(t+dt); 1] = exp(M dt)[x(t); 1].  This implementation is used as
    a numerical verification path for the DOP853 solver and is not a separate
    biological model.
    """
    from scipy.linalg import expm

    time = np.asarray(time, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing 1-D array with at least two points")
    if x0.shape != (4,):
        raise ValueError("x0 must have shape (4,)")
    A = theta.matrix_A()
    c = theta.vector_c()
    M = np.zeros((5, 5), dtype=float)
    M[:4, :4] = A
    M[:4, 4] = c
    z = np.concatenate([x0, [1.0]])
    states = np.empty((len(time), 4), dtype=float)
    states[0] = x0
    for i, dt in enumerate(np.diff(time), start=1):
        z = expm(M * float(dt)) @ z
        states[i] = z[:4]
    return Trajectory(time=time, state=states)
