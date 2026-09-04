"""
Levenberg-Marquardt, written out by hand.

scipy.optimize.least_squares would do the job, but every uncertainty number this
package prints comes out of the covariance the solver returns. If I can't say
where that covariance came from, I have no business writing "± 3 cm" next to a
measurement. So the solver is ours too.

We minimize ||r(p)||^2. Gauss-Newton solves (J^T J) dp = -J^T r; LM damps that
into (J^T J + lambda*diag(J^T J)) dp = -J^T r, so a large lambda behaves like
gradient descent and a small one like Gauss-Newton.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    p: np.ndarray            # parameters found
    cost: float              # final 0.5 * ||r||^2
    residuals: np.ndarray
    converged: bool
    steps: int
    stop_reason: str
    covariance: np.ndarray | None = None
    history: list[float] = field(default_factory=list)

    @property
    def rms(self) -> float:
        """RMS of the residuals, i.e. the error in measurement units."""
        return float(np.sqrt(np.mean(self.residuals ** 2)))


def numerical_jacobian(residual_fn, p: np.ndarray, step: float = 1e-7) -> np.ndarray:
    """Central differences. Also what we check the analytic Jacobian against."""
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
    Least-squares fit.

    residual_fn(p) gives an (m,) residual vector, jacobian_fn(p) an (m, n)
    Jacobian. Without a jacobian_fn we fall back to central differences.

    Covariance at the solution is s^2 * (J^T J)^-1 with s^2 = ||r||^2 / (m - n).
    So the noise level isn't assumed from outside, it's read off the fit residual.
    """
    p = np.asarray(p0, dtype=float).copy()
    jac = jacobian_fn if jacobian_fn is not None else (lambda q: numerical_jacobian(residual_fn, q))

    r = np.asarray(residual_fn(p), dtype=float)
    cost = 0.5 * float(r @ r)
    lam = lambda0
    history = [cost]
    reason = "max_steps"
    converged = False
    step_no = 0          # max_steps=0 never enters the loop but we still report a count

    for step_no in range(1, max_steps + 1):
        J = np.asarray(jac(p), dtype=float)
        g = J.T @ r
        if np.max(np.abs(g)) < grad_tol:
            reason, converged = "gradient", True
            break

        H = J.T @ J
        diagonal = np.diag(np.maximum(np.diag(H), 1e-12))

        # Raise the damping until we get a step that actually helps.
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
                # Worked, so ease off the damping and drift back towards Gauss-Newton.
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

    # Rebuild J here: if the last iteration accepted a step, the J we're holding
    # belongs to the previous p and the covariance would be off.
    cov = _covariance(jac(p), r) if compute_covariance else None

    return Result(
        p=p, cost=cost, residuals=r, converged=converged,
        steps=step_no, stop_reason=reason, covariance=cov, history=history,
    )


def _covariance(J: np.ndarray, r: np.ndarray) -> np.ndarray | None:
    m, n = J.shape
    dof = m - n
    if dof <= 0:
        return None                     # nothing left to estimate the noise from
    s2 = float(r @ r) / dof
    H = J.T @ J
    try:
        return s2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return s2 * np.linalg.pinv(H)
