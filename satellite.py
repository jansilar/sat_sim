from typing import List
from rk4 import rk4_step
from orbit_dynamics import orbit_derivs

class Satellite:
    def __init__(self, initial_state: List[float], initial_time: float = 0.0):
        """
        Initialize the satellite with a given state.

        Parameters:
        - initial_state: [x, y, vx, vy] in SI units (m, m, m/s, m/s)
        - initial_time: time in seconds
        """
        self.state = initial_state
        self.t = initial_time
        self.history = [(initial_time, initial_state.copy())]

    def step(self, dt: float):
        """
        Advance the simulation by one time step using RK4.

        Parameters:
        - dt: time step in seconds
        """
        self.state = rk4_step(orbit_derivs, self.state, self.t, dt)
        self.t += dt
        self.history.append((self.t, self.state.copy()))

    def simulate(self, duration: float, dt: float):
        """
        Simulate the satellite motion over a given duration.

        Parameters:
        - duration: total time to simulate in seconds
        - dt: time step in seconds
        """
        steps = int(duration / dt)
        for _ in range(steps):
            self.step(dt)

    def get_trajectory(self):
        """
        Get the full trajectory as two lists: time and state vectors.
        """
        times = [t for t, _ in self.history]
        states = [s for _, s in self.history]
        return times, states