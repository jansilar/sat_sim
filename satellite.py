from typing import List
import math
from rk4 import rk4_step
from orbit_dynamics import orbit_derivs

class Satellite:
    def __init__(self, initial_state: List[float], initial_time: float = 0.0):
        """
        Initialize the satellite with a given state.

        Parameters:
        - initial_state: ECI in 2D, [x, y, vx, vy] in SI units (m, m, m/s, m/s)
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
    
    def get_x(self) -> float:
        """
        Return the current x position of the satellite.
        """
        return self.state[0]
    
    def get_y(self) -> float:
        """
        Return the current y position of the satellite.
        """
        return self.state[1]
    
    def get_vx(self) -> float:
        """
        Return the current x velocity of the satellite.
        """
        return self.state[2]
    
    def get_vy(self) -> float:
        """
        Return the current y velocity of the satellite.
        """
        return self.state[3]
    
    def get_v(self) -> float:
        """
        Return the current speed of the satellite.
        """
        return math.sqrt(self.get_vx()**2 + self.get_vy()**2)
    
    def get_t(self) -> float:
        """
        Return the current time of the simulation.
        """
        return self.t
    
    def ground_track(self, theta_earth) -> float:
        """
        Compute the sub-satellite longitude (degrees).
        """
        x, y = self.get_x(), self.get_y()
        # ECI longitude minus Earth's rotation
        lon_rad = math.atan2(y, x) - theta_earth
        lon_deg = math.degrees(lon_rad)
        # Normalize to [-180, 180]
        lon_deg = (lon_deg + 180) % 360 - 180
        return lon_deg, 0.0  # lat is always 0 in this 2D model
    
    def get_height(self, R_EARTH = 6371e3) -> float:
        """
        Compute the height above Earth's surface.
        """
        x, y = self.get_x(), self.get_y()
        height = math.sqrt(x**2 + y**2) - R_EARTH
        return height  # return in meters