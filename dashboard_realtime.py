import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from rk4 import rk4_step
from orbit_dynamics import orbit_derivs   # uses SI units: meters, seconds
# If you prefer to use the Satellite class, you can import it and drive it instead

# --------------------
# Parameters & consts
# --------------------
R_EARTH = 6371e3                # m
GM = 3.98589196e14              # m^3/s^2
OMEGA_EARTH = 7.2921159e-5      # rad/s (sidereal)

# Simulation settings
dt = 50.0                        # simulation step (s) per frame
real_time_sleep = 0.02          # pause between frames (s) — controls animation speed
history_max = 2000              # keep last N points for drawing

# Satellite initial conditions (circular-ish)
altitude = 500e3                # 500 km
r0 = R_EARTH + altitude
v0 = math.sqrt(GM / r0)
state = [r0, 0.0, 0.0, v0]      # [x, y, vx, vy] in ECI (m, m/s)
t_sim = 0.0

# Ground station: provide longitude_deg (and latitude if 3D — here 2D so lat=0)
station_longitude_deg = 14.4378   # example: Prague approx lon
station_lon0 = math.radians(station_longitude_deg)

# Data buffers
xs, ys = [], []
lons, lats = [], []
times = []
heights = []

# Prepare figure and axes
fig = plt.figure(figsize=(14, 6))
plt.subplots_adjust(wspace=0.35)

ax_orbit = fig.add_subplot(1, 3, 1)
ax_orbit.set_aspect("equal")
ax_orbit.set_title("Orbit (ECI)")
ax_orbit.set_xlabel("x [km]")
ax_orbit.set_ylabel("y [km]")

ax_track = fig.add_subplot(1, 3, 2)
ax_track.set_title("Ground track")
ax_track.set_xlabel("Longitude [°]")
ax_track.set_ylabel("Latitude [°]")
ax_track.set_xlim([-180, 180])
ax_track.set_ylim([-90, 90])
ax_track.grid(True)

ax_height = fig.add_subplot(1, 3, 3)
ax_height.set_title("Height vs time")
ax_height.set_xlabel("Time [min]")
ax_height.set_ylabel("Height above Earth [km]")
ax_height.set_xlim([0, 60])
height_max = 1000
ax_height.set_ylim([0, height_max])
ax_height.grid(True)

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

# Auto-scaling orbit axes initially
axis_extent_orbit = (r0 * 1.2) / 1000.0  # km
ax_orbit.set_xlim(-axis_extent_orbit, axis_extent_orbit)
ax_orbit.set_ylim(-axis_extent_orbit, axis_extent_orbit)

# Helper functions
def station_position_eci(t: float, lon0_rad: float, R=R_EARTH):
    """Compute ground station position in ECI frame at time t by rotating fixed Earth coordinates."""
    theta = OMEGA_EARTH * t + lon0_rad
    x = R * math.cos(theta)
    y = R * math.sin(theta)
    return np.array([x, y])

def compute_groundtrack_and_height(sat_state, t):
    """Return (lon_deg, lat_deg, height above Earth for 2D model (lat always 0)."""
    # ECI sat pos
    sx, sy = sat_state[0], sat_state[1]
    # Convert to ECEF by rotating by theta = omega * t (ECI->ECEF)
    theta = OMEGA_EARTH * t
    x_ecef = sx * math.cos(theta) + sy * math.sin(theta)
    y_ecef = -sx * math.sin(theta) + sy * math.cos(theta)

    # lat/lon (2D: lat ~ 0)
    lat = 0.0
    lon = math.degrees(math.atan2(y_ecef, x_ecef))

    # height above Earth surface
    height = math.sqrt(x_ecef**2 + y_ecef**2) - R_EARTH

    return lon, lat, height

# Animation loop (simple real-time loop)
try:
    while True:
        # Record state
        xs.append(state[0] / 1000.0)   # km
        ys.append(state[1] / 1000.0)
        lon_deg, lat_deg, height = compute_groundtrack_and_height(state, t_sim)
        lons.append(lon_deg)
        lats.append(lat_deg)
        times.append(t_sim)
        heights.append(height/1000.0)  # km

        # Keep buffers manageable
        if len(xs) > history_max:
            xs.pop(0); ys.pop(0); lons.pop(0); lats.pop(0); times.pop(0); heights.pop(0)

        # Update artists
        orbit_line.set_data(xs, ys)
        sat_point.set_data([xs[-1]], [ys[-1]])
        track_line.set_data(lons, lats)
        track_point.set_data([lon_deg], [lat_deg])
        hight_line.set_data(np.array(times)/60.0, heights)

        # update orbit axes limits if satellite drifts out of view
        cur_x = state[0] / 1000.0
        cur_y = state[1] / 1000.0
        axis_extent_orbit = max(abs(cur_x), abs(cur_y), axis_extent_orbit)
        ax_orbit.set_xlim(-axis_extent_orbit, axis_extent_orbit)
        ax_orbit.set_ylim(-axis_extent_orbit, axis_extent_orbit)

        # Update height plot limits
        ax_height.set_xlim([0, max(times)/60.0 + 1])
        height_max = max(height/1000.0, height_max)
        ax_height.set_ylim([0, height_max])

        # Redraw a small marker for station (project station ECI to km)
        gs_eci = station_position_eci(t_sim, station_lon0)
        if gs_patch is not None:
            gs_patch.remove()
        gs_patch = Rectangle((gs_eci[0] / 1000.0, gs_eci[1] / 1000.0), 200, 200, color="blue", label="Station")
        ax_orbit.add_patch(gs_patch)

        # redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        # advance simulation
        state = rk4_step(orbit_derivs, state, t_sim, dt)
        t_sim += dt

        # keep the animation at roughly human speed
        time.sleep(real_time_sleep)

except KeyboardInterrupt:
    print("\nAnimation stopped by user.")
    plt.show()