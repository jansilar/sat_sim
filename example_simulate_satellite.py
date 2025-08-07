import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from satellite import Satellite

# Constants
EARTH_RADIUS = 6371e3  # meters
ORBIT_ALTITUDE = 400e3  # 400 km above surface
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
M_EARTH = 5.972e24  # mass of Earth, kg
GM = G * M_EARTH  # standard gravitational parameter, m^3/s^2

import math

# Compute orbital velocity for a circular orbit
r0 = EARTH_RADIUS + ORBIT_ALTITUDE
v0 = math.sqrt(GM / r0)

# Initial state: [x, y, vx, vy]
initial_state = [r0, 0, 0, v0]

# Create satellite object
sat = Satellite(initial_state)

# Simulate for 1.5 orbits, with a time step of 10 seconds
orbital_period = 2 * math.pi * r0 / v0
simulation_time = 1.5 * orbital_period
time_step = 10  # seconds

sat.simulate(simulation_time, time_step)

# Retrieve trajectory
times, states = sat.get_trajectory()
xs = [s[0] / 1000 for s in states]  # convert to km for plotting
ys = [s[1] / 1000 for s in states]

# Plot trajectory
plt.figure(figsize=(8, 8))
plt.plot(0, 0, 'o', label="Earth", color="blue")
plt.plot(initial_state[0]/1000,initial_state[1]/1000, 'o', label="Satellite init", color="red")
plt.plot(xs, ys, label="Satellite trajectory")

plt.axis("equal")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.title("Satellite Orbit Around Earth (2D Simulation)")
plt.legend()
plt.grid(True)
plt.show()