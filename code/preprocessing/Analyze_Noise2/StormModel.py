"""This class is used to encapsulate the attributes for storms."""
import datetime

class StormModel:
    """Initialization function."""

    def __init__(self, sid, time, lat, lon):
        """Intialization function for StormModel.

        :param sid: .
        :param season: 
        ...
        """
        self.sid = sid
        self.time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        self.lat = float(lat)
        self.lon = float(lon)
        self.calLat = 0.0
        self.calLon = 0.0

    def __repr__(self) -> str:
        """Represent the content of storm."""
        return f'{self.sid} - {self.time} - "org: {self.lat} - {self.lon}" "cal: {self.calLat} - {self.calLon}"'