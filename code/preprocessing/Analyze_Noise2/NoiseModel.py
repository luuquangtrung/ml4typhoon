import datetime


class NoiseModel():
    """IngestorInterface class definition."""

    #trackingDate = datetime.datetime.now()
    trackingArea = {
        "min_lat": 0,
        "max_lat": 0,
        "min_lon": 0,
        "max_lon": 0,
    }
    actualNoiseFlag = "NOT_NEAR" # 0x00: NOT_NEAR, 0x01: NEAR, 0x10: IN
    expectNoiseFlag = "NOT_NEAR"
    impactStorms = []
    passed = True
    impactStormsStr = ""
    distanceInDegree = ""
    distanceInKm = ""

    def __repr__(self) -> str:
        """Represent the content of storm."""
        str = f'actual: {self.actualNoiseFlag} - expected: {self.expectNoiseFlag}'
        str += f' - min_lat: {self.trackingArea["min_lat"]} - max_lat: {self.trackingArea["max_lat"]} - min_lon: {self.trackingArea["min_lon"]} - max_lon: {self.trackingArea["max_lon"]}'
        return str
    
    def reset(self):
        self.trackingDate = datetime.datetime.now()
        self.trackingArea = {
            "min_lat": 0,
            "max_lat": 0,
            "min_lon": 0,
            "max_lon": 0,
        }
        self.actualNoiseFlag = "NOT_NEAR"
        self.expectNoiseFlag = "NOT_NEAR"
        self.impactStorms.clear()
        self.passed = True

    def calculateExpectNoiseFlag(self):
        for impactStorm in self.impactStorms:
            # self.impactStormsStr += repr(impactStorm)
            # self.impactStormsStr += "\n"
            if impactStorm.calLat >= self.trackingArea["min_lat"] and impactStorm.calLat <= self.trackingArea["max_lat"] and impactStorm.calLon >= self.trackingArea["min_lon"] and impactStorm.calLon <= self.trackingArea["max_lon"]:
                self.expectNoiseFlag = "IN"
                return
            if impactStorm.calLat >= self.trackingArea["min_lat"] - 2.0 and impactStorm.calLat <= self.trackingArea["max_lat"] + 2.0 and impactStorm.calLon >= self.trackingArea["min_lon"] - 2.0 and impactStorm.calLon <= self.trackingArea["max_lon"] + 2.0:
                self.expectNoiseFlag = "NEAR"