#import pytest
from rk4 import rk4_step
import plotly.graph_objects as go


def free_fall():
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

    tList = [t]
    xList = [x0]
    vList = [v0]

    for _ in range(50):  # t = 1.0 s
        state = rk4_step(deriv, state, t, dt)
        t += dt
        tList.append(t)
        xList.append(state[0])
        vList.append(state[1])
        print(f"t={t:.1f}, x={state[0]:.2f}, v={state[1]:.2f}")

    x_num, v_num = state
    print(str(t),", ", str(x_num),", ", str(v_num))

    fig = go.Figure(data=go.Scatter(x=tList, y=xList, mode='lines+markers', name='x vs t'))
    fig.add_trace(go.Scatter(x=tList, y=vList, mode='lines+markers', name='v vs t'))
    fig.update_layout(title='Free Fall Simulation', xaxis_title='Time (s)', yaxis_title='Position (m) / Velocity (m/s)')
    fig.show()


def main():
    free_fall()

if __name__ == "__main__":
    main()