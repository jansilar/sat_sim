State = list[float]

def orbit_derivs(state: State, t: float) -> State:
    """
    Compute the derivatives for the orbit dynamics.

    Parameters:
    -----------
    state : list of floats
        Current state vector [x, y, vx, vy]
    t : float
        Current time (not used in this example)

    Returns:
    --------
    derivatives : list of floats
        Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    G = 6.67430e-11  # Gravitational constant [m3/(kg·s2)]
    M = 5.972e24     # Mass of Earth [kg]

    x, y, vx, vy = state
    r = (x**2 + y**2)**0.5  # Distance from the center of the Earth

    # Gravitational acceleration
    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]