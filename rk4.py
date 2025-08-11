"""
Runge-Kutta 4th order (RK4) integrator for systems of ordinary differential equations.
"""

from typing import Callable, List

# State type: (e.g. [x, y, vx, vy])
State = List[float]
Input = List[float]
Params = List[float]

def rk4_step(
    derivs: Callable[[State, Input, float], State],
    state: State,
    input: Input,
    params: Params,
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
    k1 = derivs(state, input, params, t)
    k2 = derivs([s + 0.5 * dt * k for s, k in zip(state, k1)], input, params, t + 0.5 * dt)
    k3 = derivs([s + 0.5 * dt * k for s, k in zip(state, k2)], input, params, t + 0.5 * dt)
    k4 = derivs([s + dt * k for s, k in zip(state, k3)], input, params, t + dt)

    new_state = [
        s + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    ]
    return new_state