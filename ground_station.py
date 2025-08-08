import math

EARTH_RADIUS = 6371e3  # in meters
OMEGA_EARTH = 7.2921159e-5  # rad/s

class GroundStation2D:
    def __init__(self, longitude_deg: float):
        self.longitude_rad = math.radians(longitude_deg)

    def position(self, t: float):
        """
        Returns the 2D ECI position of the ground station at time t (in seconds).
        The ground station rotates with the Earth.

        Returns:
        - [x, y]: coordinates in meters
        """
        theta = OMEGA_EARTH * t + self.longitude_rad
        x = EARTH_RADIUS * math.cos(theta)
        y = EARTH_RADIUS * math.sin(theta)
        return [x, y]
    
    def isSatelliteVisible(self, satellite_position: list[float], t: float) -> bool:
        """
        Check if the satellite is visible from the ground station at time t.

        Parameters:
        - satellite_position: [x, y] coordinates of the satellite in meters
        - t: time in seconds

        Returns:
        - True if the satellite is visible, i.e. above the horizon, False otherwise
        """
        station_pos = self.position(t)
        dx = satellite_position[0] - station_pos[0]
        dy = satellite_position[1] - station_pos[1]
        dot_product = dx * station_pos[0] + dy * station_pos[1]
        return dot_product > 0