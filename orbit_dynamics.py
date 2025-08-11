State = list[float]
Input = list[float]
Params = list[float]

def orbit_derivs(state: State, input: Input, params: Params, t: float) -> State:
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
    throttle = input[0] if input else 0.0 # Throttle input
    mass = params[0] if params else 0.0  # Mass of the satellite
    max_engine_power = params[1] if len(params) > 1 else 1.0  # Max engine power
   
    r = (x**2 + y**2)**0.5  # Distance from the center of the Earth
    v = (vx**2 + vy**2)**0.5  # Speed

    a_engine = throttle * max_engine_power / (v * mass) if v > 0 else 0  # Engine acceleration

    # Gravitational + engine acceleration
    ax = -G * M * x / r**3 + (vx/v * a_engine if v > 0 else 0.0)
    ay = -G * M * y / r**3 + (vy/v * a_engine if v > 0 else 0.0)

    return [vx, vy, ax, ay]