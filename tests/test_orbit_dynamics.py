from orbit_dynamics import orbit_derivs

def test_orbit_derivs():
    """
    Simple test of the orbit_derivs function.
    """
    # Circular orbit test: radius = 7000 km, speed ~7.546 km/s
    # Expected acceleration magnitude ~ GM / r^2
    state = [7000.0, 0.0, 0.0, 7.5460491]
    t = 0.0
    deriv = orbit_derivs(state, [], [], t)
    MG = 6.67430e-11 * 5.972e24
    expected_acc_mag = MG / (7000.0**2)
    acc_mag = (deriv[2]**2 + deriv[3]**2)**0.5

    assert abs(acc_mag - expected_acc_mag) < 1e-6, f"Acceleration magnitude incorrect: {acc_mag} vs {expected_acc_mag}"

    print("Test passed: orbit_derivs computes correct acceleration for circular orbit.")