import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from rk4 import rk4_step
from satellite import Satellite
from ground_station import GroundStation2D

# --------------------
# Parameters & consts
# --------------------
R_EARTH = 6371e3                # m
GM = 3.98589196e14              # m^3/s^2
OMEGA_EARTH = 7.2921159e-5      # rad/s (sidereal)

# Simulation settings
dt = 60.0                        # simulation step (s) per frame
real_time_sleep = 0.02          # pause between frames (s) — controls animation speed
history_max = 2000              # keep last N points for drawing

# Satellite initial conditions (circular-ish)
altitude = 500e3                # 500 km
r0 = R_EARTH + altitude
v0 = math.sqrt(GM / r0)
state = [r0, 0.0, 0.0, v0]      # [x, y, vx, vy] in ECI (m, m/s)
t_sim = 0.0

throttle = 0.0  # 0 = off, 1 = max power

max_engine_power = 100e3 # W, maximal engine power
m_satellite = 500.0  # kg, mass of the satellite

satellite = Satellite(initial_state=state, initial_time=t_sim, m=m_satellite, max_engine_power=max_engine_power)

# Ground station: provide longitude_deg (and latitude if 3D — here 2D so lat=0)
station_longitude_deg = 14.4378   # example: Prague approx lon
station_lon0 = math.radians(station_longitude_deg)

ground_station = GroundStation2D(longitude_deg=station_longitude_deg)

# Data buffers
xs, ys = [], []
lons, lats = [], []
times = []
heights = []
velocities = []
throttles = []

# Prepare figure and axes
fig = plt.figure(figsize=(14, 6))
plt.subplots_adjust(wspace=0.35)

ax_orbit = fig.add_subplot(2, 2, 1)
ax_orbit.set_aspect("equal")
ax_orbit.set_title("Orbit (ECI)")
ax_orbit.set_xlabel("x [km]")
ax_orbit.set_ylabel("y [km]")

ax_track = fig.add_subplot(2, 2, 2)
ax_track.set_title("Ground track")
ax_track.set_xlabel("Longitude [°]")
ax_track.set_ylabel("Latitude [°]")
ax_track.set_xlim([-180, 180])
ax_track.set_ylim([-90, 90])
ax_track.grid(True)

ax_height = fig.add_subplot(2, 2, 3)
ax_height.set_title("Height vs time")
ax_height.set_xlabel("Time [min]")
ax_height.set_ylabel("Height above Earth [km]")
ax_height.set_xlim([0, 60])
height_max = 1000
ax_height.set_ylim([0, height_max])
ax_height.grid(True)

# Add secondary y-axis for velocity
ax_velocity = ax_height.twinx()
ax_velocity.set_ylabel("Velocity [km/s]")
velocity_max = 10
ax_velocity.set_ylim([0, velocity_max])

ax_throttle = fig.add_subplot(2, 2, 4)
ax_throttle.set_title("Engine throttle vs time")
ax_throttle.set_xlabel("Time [min]")
ax_throttle.set_ylabel("Engine throttle [1]")
ax_throttle.set_xlim([0, 60])
ax_throttle.set_ylim([-1.02, 1.02])
ax_throttle.grid(True)



# Plot the Earth circle in orbit view (in km for nicer scale)
earth_patch = Circle((0, 0), R_EARTH / 1000.0, color="lightblue", zorder=0)
ax_orbit.add_patch(earth_patch)
gs_patch = None  # Ground station patch


# Artists we will update
orbit_line, = ax_orbit.plot([], [], "r-", lw=1)
sat_point, = ax_orbit.plot([], [], "ro", ms=6)
track_line, = ax_track.plot([], [], "g-", lw=1)
track_point, = ax_track.plot([], [], "gx", ms=6)
hight_line, = ax_height.plot([], [], "b-")
velocity_line, = ax_velocity.plot([], [], "r-")
throtte_line, = ax_throttle.plot([], [], "m-")

# Auto-scaling orbit axes initially
axis_extent_orbit = (r0 * 1.2) / 1000.0  # km
ax_orbit.set_xlim(-axis_extent_orbit, axis_extent_orbit)
ax_orbit.set_ylim(-axis_extent_orbit, axis_extent_orbit)

# Add control instructions as a text note below the throttle plot
ax_throttle.text(
    0.5, 0.1,
    "Throttle: up arrow = increase, down arrow = decrease",
    ha="center", va="top", fontsize=10, color="black", fontweight="bold", transform=ax_throttle.transAxes
)

def on_key(event):
    global throttle
    if event.key == "up":
        throttle = min(1.0, throttle + 0.1)  # increase throttle within limits
    elif event.key == "down":
        throttle = max(-1.0, throttle - 0.1)  # decrease throttle
    elif event.key == "space":
        throttle = 0.0   # reset throttle
    print(f"Throttle: {throttle:.2f} m/s^2")

fig.canvas.mpl_connect("key_press_event", on_key)

# Animation loop (simple real-time loop)
try:
    while True:
        # Record state
        xs.append(satellite.get_x() / 1000.0)   # km
        ys.append(satellite.get_y() / 1000.0)
        lon_deg, lat_deg = satellite.ground_track(OMEGA_EARTH * t_sim)
        height = satellite.get_height()  # in meters
        lons.append(lon_deg)
        lats.append(lat_deg)
        times.append(t_sim)
        heights.append(height/1000.0)  # km
        velocities.append(satellite.get_v() / 1000.0)  # km/s
        throttles.append(throttle)

        # Keep buffers manageable
        if len(xs) > history_max:
            xs.pop(0)
            ys.pop(0)
            lons.pop(0)
            lats.pop(0)
            times.pop(0)
            heights.pop(0)
            velocities.pop(0)
            throttles.pop(0)

        # Update artists
        orbit_line.set_data(xs, ys)
        sat_point.set_data([xs[-1]], [ys[-1]])
        track_line.set_data(lons, lats)
        track_point.set_data([lon_deg], [lat_deg])
        hight_line.set_data(np.array(times)/60.0, heights)
        velocity_line.set_data(np.array(times)/60.0, velocities)
        throtte_line.set_data(np.array(times)/60.0, throttles)

        # update orbit axes limits if satellite drifts out of view
        cur_x = satellite.get_x() / 1000.0
        cur_y = satellite.get_y() / 1000.0
        axis_extent_orbit = max(abs(cur_x), abs(cur_y), axis_extent_orbit)
        ax_orbit.set_xlim(-axis_extent_orbit, axis_extent_orbit)
        ax_orbit.set_ylim(-axis_extent_orbit, axis_extent_orbit)

        # Update height plot limits
        ax_height.set_xlim([times[0]/60, t_sim/60.0 + 1])
        height_max = max(height/1000.0, height_max)
        ax_height.set_ylim([0, height_max])

        # Update velocity plot limits
        velocity_max = max(satellite.get_v() / 1000.0, velocity_max)
        ax_velocity.set_ylim([0, velocity_max])

        # Update throttle plot limits
        ax_throttle.set_xlim([times[0]/60, t_sim/60.0 + 1])

        # Redraw a small marker for station (project station ECI to km)
        gs_eci = ground_station.position(OMEGA_EARTH * t_sim)
        if gs_patch is not None:
            gs_patch.remove()
        gs_patch = Rectangle((gs_eci[0] / 1000.0, gs_eci[1] / 1000.0), 200, 200, color="blue", label="Station")
        ax_orbit.add_patch(gs_patch)

        # redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        if not satellite.is_above_ground(R_EARTH):
            fig.text(0.5, 0.5, "SATELLITE HAS HIT THE GROUND!", color="red", fontsize=24, ha="center", va="center", zorder=100)
            fig.canvas.draw()
            #plt.pause(10)
            plt.waitforbuttonpress()  # waits until key or mouse button pressed
            break
        plt.pause(0.001)

        # advance simulation
        satellite.step(dt, throttle)
        t_sim += dt

        # keep the animation at roughly human speed
        time.sleep(real_time_sleep)

    

except KeyboardInterrupt:
    print("\nAnimation stopped by user.")
    plt.show()