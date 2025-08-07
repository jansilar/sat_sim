#import pytest
from rk4 import rk4_step


def test_free_fall():
    # dx/dt = v, dv/dt = -g
    def deriv(state, t):
        x, v = state
        g = -9.81
        return [v, g]

    x0 = 100.0
    v0 = 0.0
    t = 0.0
    dt = 0.1
    state = [x0, v0]

    for _ in range(10):  # t = 1.0 s
        state = rk4_step(deriv, state, t, dt)
        t += dt

    x_num, v_num = state
    x_exact = x0 + v0 * t + 0.5 * (-9.81) * t**2
    v_exact = v0 + (-9.81) * t

    assert abs(x_num - x_exact) < 1e-2
    assert abs(v_num - v_exact) < 1e-2


