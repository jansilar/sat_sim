"""
Runge-Kutta 4th order (RK4) integrator for systems of ordinary differential equations.
"""

from typing import Callable, List, Union

# State type: (e.g. [x, y, vx, vy])
State = Union[List[float], tuple[float, ...]]

def rk4_step(
    derivs: Callable[[State, float], State],
    state: State,
    t: float,
    dt: float
) -> State:
    """
    Perform a single RK4 integration step.

    Parameters:
    -----------
    derivs : callable
        Function computing the derivatives (dy/dt) at given state and time: f(state, t)
    state : list or tuple of floats
        Current state vector
    t : float
        Current time
    dt : float
        Time step

    Returns:
    --------
    new_state : list of floats
        State after time step dt
    """
    k1 = derivs(state, t)
    k2 = derivs([s + 0.5 * dt * k for s, k in zip(state, k1)], t + 0.5 * dt)
    k3 = derivs([s + 0.5 * dt * k for s, k in zip(state, k2)], t + 0.5 * dt)
    k4 = derivs([s + dt * k for s, k in zip(state, k3)], t + dt)

    new_state = [
        s + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    ]
    return new_state