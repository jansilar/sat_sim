import math

class GroundStation2D:
    def __init__(self, longitude_deg: float):
        self.longitude_rad = math.radians(longitude_deg)

    def position(self, theta_earth: float, EARTH_RADIUS=6371e3):
        """
        Returns the 2D ECI position of the ground station given Earth rotation.
        The ground station rotates with the Earth.

        Parameters:
        - theta_earth: Earth rotation angle in radians
        - EARTH_RADIUS: radius of the Earth in meters (default: 6371e3)

        Returns:
        - [x, y]: coordinates in meters
        """
        theta = theta_earth + self.longitude_rad
        x = EARTH_RADIUS * math.cos(theta)
        y = EARTH_RADIUS * math.sin(theta)
        return [x, y]
    
    def is_satellite_visible(self, satellite_position: list[float], theta_earth: float) -> bool:
        """
        Check if the satellite is visible from the ground station given Earth rotation.

        Parameters:
        - satellite_position: [x, y] coordinates of the satellite in meters
        - earth rotation angle theta_earth in radians

        Returns:
        - True if the satellite is visible, i.e. above the horizon, False otherwise
        """
        station_pos = self.position(theta_earth)
        dx = satellite_position[0] - station_pos[0]
        dy = satellite_position[1] - station_pos[1]
        dot_product = dx * station_pos[0] + dy * station_pos[1]
        return dot_product > 0