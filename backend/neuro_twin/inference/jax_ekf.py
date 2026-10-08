"""JAX autodiff EKF likelihood for configurable nonlinear observations and persistent random effects."""
from __future__ import annotations

import numpy as np

from neuro_twin.core.parameters import PARAMETER_NAMES

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    expm = None
    JAX_AVAILABLE = False


def require_jax():
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")


def theta_A_c(theta):
    t = jnp.asarray(theta)
    aP,bPI,dP,bIP,dI,bNP,bNI,dN,bQN,bQI,dQ = t
    A = jnp.array([[-dP,bPI,0.,0.],[bIP,-dI,0.,0.],[bNP,bNI,-dN,0.],[0.,bQI,bQN,-dQ]], dtype=t.dtype)
    c = jnp.array([aP,0.,0.,0.], dtype=t.dtype)
    return A,c


def affine_transition(theta, dt):
    A,c = theta_A_c(theta)
    M = jnp.zeros((5,5), dtype=jnp.float64)
    M = M.at[:4,:4].set(A)
    M = M.at[:4,4].set(c)
    E = expm(M*dt)
    return E[:4,:4], E[:4,4]


def _link(name, eta):
    if name == "identity": return eta
    if name == "exp": return jnp.exp(eta)
    if name == "log": return jnp.log(jnp.maximum(eta, 1e-12))
    if name == "softplus": return jax.nn.softplus(eta)
    if name == "sigmoid": return jax.nn.sigmoid(eta)
    raise ValueError(name)


def _predict(specs, state, theta):
    vals=[]
    for spec in specs:
        wx=jnp.asarray(spec["state_weights"])
        wt=jnp.asarray(spec["parameter_weights"])
        eta=wx@state+wt@theta+spec["bias"]
        vals.append(_link(spec["link"], eta))
    return jnp.stack(vals)


def ekf_loglik_jax(time, observations, specs, R, initial_mean, initial_covariance, theta, *, Qc=None, Z=None, random_effect_covariance=None):
    require_jax()
    t=jnp.asarray(time,dtype=jnp.float64)
    Y=jnp.asarray(observations,dtype=jnp.float64)
    R=jnp.asarray(R,dtype=jnp.float64)
    x0=jnp.asarray(initial_mean,dtype=jnp.float64)
    P0=jnp.asarray(initial_covariance,dtype=jnp.float64)
    theta=jnp.asarray(theta,dtype=jnp.float64)
    Qc=jnp.zeros((4,4),dtype=jnp.float64) if Qc is None else jnp.asarray(Qc,dtype=jnp.float64)
    Zm=None if Z is None else jnp.asarray(Z,dtype=jnp.float64)
    n_u=0 if Zm is None else Zm.shape[1]
    n=4+n_u
    Iaug=jnp.eye(n)
    if Zm is not None:
        if random_effect_covariance is None:
            raise ValueError("random_effect_covariance is required when Z is provided")
        U0=jnp.asarray(random_effect_covariance,dtype=jnp.float64)
        if U0.shape != (n_u,n_u):
            raise ValueError("random_effect_covariance shape mismatch")
        x0=jnp.concatenate([x0,jnp.zeros((n_u,),dtype=jnp.float64)])
        P0=jnp.block([[P0,jnp.zeros((4,n_u),dtype=jnp.float64)],
                      [jnp.zeros((n_u,4),dtype=jnp.float64),U0]])

    def h_all(xaug):
        base = _predict(specs, xaug[:4], theta)
        if Zm is None:
            return base
        return base + Zm @ xaug[4:]

    def h_jac(xaug):
        return jax.jacfwd(h_all)(xaug)

    def step(carry, data):
        m,P,ll=carry
        k,tk,yk=data
        def do_prop(_):
            phi,off=affine_transition(theta, tk-t[k-1])
            F=Iaug.at[:4,:4].set(phi)
            q=Iaug*0.0
            q=q.at[:4,:4].set(Qc)
            return F@m + jnp.concatenate([off,jnp.zeros((n_u,))]), F@P@F.T+q
        mp,Pp=jax.lax.cond(k==0, lambda _: (m,P), do_prop, operand=None)
        mask=jnp.isfinite(yk)
        yclean=jnp.where(mask,yk,0.0)
        H=h_jac(mp)
        hh=h_all(mp)
        D=jnp.diag(mask.astype(jnp.float64))
        Ho=D@H
        Rm=D@R@D + jnp.diag((~mask).astype(jnp.float64)*1e12)
        innov=jnp.where(mask, yclean-hh, 0.0)
        S=Ho@Pp@Ho.T+Rm
        S=0.5*(S+S.T)
        L=jnp.linalg.cholesky(S)
        a=jax.scipy.linalg.solve_triangular(L, innov, lower=True)
        Sinvinnov=jax.scipy.linalg.solve_triangular(L.T,a,lower=False)
        Iinv=jax.scipy.linalg.solve_triangular(L, jnp.eye(H.shape[0]), lower=True)
        Sinv=jax.scipy.linalg.solve_triangular(L.T,Iinv,lower=False)
        K=Pp@Ho.T@Sinv
        mn=mp+K@innov
        J=Iaug-K@Ho
        Pn=0.5*(J@Pp@J.T + K@Rm@K.T + (J@Pp@J.T + K@Rm@K.T).T)
        logdet=2*jnp.sum(jnp.log(jnp.diag(L)))
        nobs=jnp.sum(mask)
        nmiss=H.shape[0]-nobs
        maha=innov@Sinvinnov
        constant_missing=nmiss*(jnp.log(2*jnp.pi)+jnp.log(1e12))
        inc=-0.5*(H.shape[0]*jnp.log(2*jnp.pi)+logdet+maha-constant_missing)
        return (mn,Pn,ll+inc),inc

    ks=jnp.arange(t.shape[0],dtype=jnp.int32)
    (mf,Pf,ll),inc=jax.lax.scan(step,(x0,P0,0.0),(ks,t,Y))
    return ll


def value_and_grad_fn(time, observations, specs, R, initial_covariance, *, Qc=None, Z=None):
    require_jax()
    def obj(z):
        return -ekf_loglik_jax(time, observations, specs, R, z[:4], initial_covariance, z[4:], Qc=Qc, Z=Z)
    vg=jax.jit(jax.value_and_grad(obj))
    def wrapped(z_np):
        v,g=vg(jnp.asarray(z_np,dtype=jnp.float64))
        return float(v),np.asarray(g,float)
    return wrapped
