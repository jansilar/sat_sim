from ground_station import GroundStation2D
import math

def test_GroundStation2D():
    """
    Test the GroundStation2D class for correct position and visibility.
    """
    longitude_deg = 30.0
    ground_station = GroundStation2D(longitude_deg)
    
    # Test position at t=0
    t = 0.0
    position = ground_station.position(t)
    expected_x = 6371e3 * math.cos(math.radians(longitude_deg))
    expected_y = 6371e3 * math.sin(math.radians(longitude_deg))
    
    assert abs(position[0] - expected_x) < 1e-6, f"X position incorrect: {position[0]} vs {expected_x}"
    assert abs(position[1] - expected_y) < 1e-6, f"Y position incorrect: {position[1]} vs {expected_y}"
    
    # Test visibility with a satellite above the horizon
    satellite_position = [expected_x + 1000, expected_y + 1000]  # Satellite slightly above the ground station
    assert ground_station.isSatelliteVisible(satellite_position, t), "Satellite should be visible"
    
    # Test visibility with a satellite below the horizon
    satellite_position = [expected_x - 1000, expected_y - 1000]  # Satellite below the ground station
    assert not ground_station.isSatelliteVisible(satellite_position, t), "Satellite should not be visible"

    print("Test passed: GroundStation2D computes correct position and visibility.")