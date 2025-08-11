"""
Runge-Kutta 4th order (RK4) integrator for systems of ordinary differential equations.
"""

from typing import Callable, List

# State type: (e.g. [x, y, vx, vy])
Vector = List[float]

def rk4_step(
    derivs: Callable[[Vector, Vector, Vector, float], Vector],
    state: Vector,
    input: Vector,
    params: Vector,
    t: float,
    dt: float
) -> Vector:
    """
    Perform a single RK4 integration step.

    Parameters
    ----------
    derivs : Callable[[Vector, Vector, Vector, float], State]
        Function computing the derivatives (dy/dt) at given state, input, params and time: f(state, input, params, t).
    state : list of float
        Current state vector.
    input : list of float
        Input vector (e.g. throttle).
    params : list of float
        Additional parameters (e.g. mass, max engine power).
    t : float
        Current time.
    dt : float
        Time step.

    Returns
    -------
    list of float
        State after time step dt.
    """
    k1 = derivs(state, input, params, t)
    k2 = derivs([s + 0.5 * dt * k for s, k in zip(state, k1)], input, params, t + 0.5 * dt)
    k3 = derivs([s + 0.5 * dt * k for s, k in zip(state, k2)], input, params, t + 0.5 * dt)
    k4 = derivs([s + dt * k for s, k in zip(state, k3)], input, params, t + dt)

    new_state = [
        s + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    ]
    return new_state