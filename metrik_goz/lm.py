"""
Levenberg-Marquardt — a hand-written damped least-squares solver.

Why not the ready-made `scipy.optimize.least_squares`: every uncertainty claim
this package makes rests on the covariance the solver produces. If we don't
know where that covariance comes from, we have no right to say "±3 cm". So the
solver is ours too.

The problem solved:
    min_p  ||r(p)||^2
The Gauss-Newton step solves (J^T J) dp = -J^T r; LM turns that into
(J^T J + lambda * diag(J^T J)) dp = -J^T r. A large lambda makes the step
approach gradient descent, a small one makes it approach Gauss-Newton.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    """The outcome of an LM solve."""

    p: np.ndarray                      # the parameters found
    cost: float                        # final 0.5 * ||r||^2
    residuals: np.ndarray              # final residual vector
    converged: bool
    steps: int
    stop_reason: str
    covariance: np.ndarray | None = None   # covariance of the parameters
    history: list[float] = field(default_factory=list)

    @property
    def rms(self) -> float:
        """Root mean square of the residuals — the error in measurement units."""
        return float(np.sqrt(np.mean(self.residuals ** 2)))


def numerical_jacobian(residual_fn, p: np.ndarray, step: float = 1e-7) -> np.ndarray:
    """Central-difference Jacobian. Also used to verify the analytic Jacobian."""
    p = np.asarray(p, dtype=float)
    r0 = np.asarray(residual_fn(p), dtype=float)
    J = np.zeros((r0.size, p.size))
    for i in range(p.size):
        h = step * max(1.0, abs(p[i]))
        forward, backward = p.copy(), p.copy()
        forward[i] += h
        backward[i] -= h
        J[:, i] = (np.asarray(residual_fn(forward)) - np.asarray(residual_fn(backward))) / (2 * h)
    return J


def solve(
    residual_fn,
    p0,
    jacobian_fn=None,
    *,
    max_steps: int = 100,
    lambda0: float = 1e-3,
    cost_tol: float = 1e-12,
    step_tol: float = 1e-12,
    grad_tol: float = 1e-12,
    compute_covariance: bool = True,
) -> Result:
    """
    Minimizes the residual function in the least-squares sense.

    residual_fn(p) -> (m,) residual vector
    jacobian_fn(p) -> (m, n) Jacobian; if omitted, central differences are used.

    The covariance is estimated at the point of convergence as  s^2 * (J^T J)^-1,
    where s^2 = ||r||^2 / (m - n) is the residual variance. That is, instead of
    assuming the measurement noise from outside, we read it off the fit residual.
    """
    p = np.asarray(p0, dtype=float).copy()
    jac = jacobian_fn if jacobian_fn is not None else (lambda q: numerical_jacobian(residual_fn, q))

    r = np.asarray(residual_fn(p), dtype=float)
    cost = 0.5 * float(r @ r)
    lam = lambda0
    history = [cost]
    reason = "max_steps"
    converged = False
    step_no = 0          # with max_steps=0 the loop never runs; we still report it

    for step_no in range(1, max_steps + 1):
        J = np.asarray(jac(p), dtype=float)
        g = J.T @ r                       # gradient
        if np.max(np.abs(g)) < grad_tol:
            reason, converged = "gradient", True
            break

        H = J.T @ J
        diagonal = np.diag(np.maximum(np.diag(H), 1e-12))

        # Keep raising the damping until an acceptable step is found.
        accepted = False
        for _ in range(30):
            try:
                dp = np.linalg.solve(H + lam * diagonal, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            p_new = p + dp
            r_new = np.asarray(residual_fn(p_new), dtype=float)
            cost_new = 0.5 * float(r_new @ r_new)

            if cost_new < cost:
                # The step worked: relax the damping, move towards Gauss-Newton.
                decrease = cost - cost_new
                p, r, cost = p_new, r_new, cost_new
                lam = max(lam * 0.3, 1e-12)
                accepted = True
                history.append(cost)
                if decrease < cost_tol or np.linalg.norm(dp) < step_tol:
                    reason, converged = "cost" if decrease < cost_tol else "step", True
                break

            lam *= 10.0

        if not accepted:
            reason, converged = "damping_saturated", True
            break
        if converged:
            break

    # The covariance needs the Jacobian evaluated at the final parameters. If the
    # loop accepted a step, the J we hold belongs to the previous p, so we rebuild.
    cov = _covariance(jac(p), r) if compute_covariance else None

    return Result(
        p=p, cost=cost, residuals=r, converged=converged,
        steps=step_no, stop_reason=reason, covariance=cov, history=history,
    )


def _covariance(J: np.ndarray, r: np.ndarray) -> np.ndarray | None:
    """s^2 (J^T J)^-1. Returns None when there are no degrees of freedom left."""
    m, n = J.shape
    dof = m - n
    if dof <= 0:
        return None
    s2 = float(r @ r) / dof
    H = J.T @ J
    try:
        return s2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return s2 * np.linalg.pinv(H)
