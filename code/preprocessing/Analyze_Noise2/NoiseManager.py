from CSVHandler import CSVHandler
from StormManager import StormManager
from NoiseModel import NoiseModel
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from math import pi

def FindNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def degreeToKm(distance):
    return round(distance / 180 * pi * 6371, 4)

class NoiseManager:
    """Initialization function."""

    def __init__(self, T:str):
        """Intialization function for NoiseManager.

        :param sid: .
        :param season: 
        ...
        """

        self._rawNoises = [] #raw data get from csv file
        self._noises = [] #data after processing raw data
        self._stormManager = StormManager()
        if T=="ncep-fnl":
            self._wd = xr.load_dataset(filename_or_obj = "../data/temp/ncep-fnl/fnl_19990730_18_00.grib1.nc", engine = "netcdf4")
        if T=="nasa-merra2":
            self._wd = xr.load_dataset(filename_or_obj = "../data/temp/nasa-merra2/merra2_19800101_00_00.nc", engine = "netcdf4")

    def loadRawNoises(self, path: str):
        self._rawNoises = CSVHandler.parseNoise(path)

    def reset(self):
        self._rawNoises.clear()
        self._noises.clear()

    def processRawNoises(self, timeScale):
        for rawNoise in self._rawNoises:
            filepath = "." + rawNoise.storm_filepath 

            noise = NoiseModel()
            noise.reset()
            noise.actualNoiseFlag = rawNoise.position

            wd = xr.load_dataset(filename_or_obj = filepath, engine = "netcdf4")
            wd_lat_grid = wd["latitude"].values
            wd_lon_grid = wd["longitude"].values
            noise.trackingArea = {
                "min_lat": np.min(wd_lat_grid),
                "max_lat": np.max(wd_lat_grid),
                "min_lon": np.min(wd_lon_grid),
                "max_lon": np.max(wd_lon_grid),
            }

            path = rawNoise.storm_filepath.split("/")
            if path[4] == "FixedDomain":
                date = path[5].split(".")[0].split("_")[2] # extract date from file name: yyyymmdd
                hour = path[5].split(".")[0].split("_")[3] # extract hour from file name: hh
                noise.trackingDate = datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(hour)) # convert to Date object                

            if path[4] == "PastDomain": 
                tmp = path[5].split(".")[0] #remove .nc
                subtractTime = tmp.split("_")[2] # extract subtract time from file name
                currentStorm = self._stormManager.getStormBySid(tmp.split("_")[1])
                noise.trackingDate = currentStorm.time - timedelta(hours=int(subtractTime) * timeScale) # time scale = 3 for merra, 6 for fnl 
                                
            if path[4] == "DynamicDomain": 
                tmp = path[5].split(".")[0] #remove .nc
                currentStorm = self._stormManager.getStormBySid(tmp.split("_")[1])
                noise.trackingDate = currentStorm.time

            # get impacted storm
            noise.impactStorms.clear()
            wd_lat_grid_ext = self._wd["latitude"].values
            wd_lon_grid_ext = self._wd["longitude"].values           

            if "-" in rawNoise.impactStorm :
                impactStorms = rawNoise.impactStorm.split("-")
                distances = []
                for impact in impactStorms:
                    storm = self._stormManager.getStormBySidAndTime(impact.split("@")[0], noise.trackingDate)
                    distances.append(float(impact.split("@")[2]))
                    if storm != None:
                        storm.calLat = FindNearest(wd_lat_grid_ext, storm.lat)
                        storm.calLon = FindNearest(wd_lon_grid_ext, storm.lon)
                        noise.impactStorms.append(storm)
                noise.distanceInDegree = "-".join(str(distance) for distance in distances)
                noise.distanceInKm = "-".join(str(degreeToKm(distance)) for distance in distances)
            else:
                storm = self._stormManager.getStormBySidAndTime(rawNoise.impactStorm.split("@")[0], noise.trackingDate)
                noise.distanceInDegree = str(rawNoise.impactStorm.split("@")[2])
                noise.distanceInKm = str(degreeToKm(float(noise.distanceInDegree)))
                if storm != None:
                    storm.calLat = FindNearest(wd_lat_grid_ext, storm.lat)
                    storm.calLon = FindNearest(wd_lon_grid_ext, storm.lon)
                    noise.impactStorms.append(storm)

            noise.calculateExpectNoiseFlag()
            noise.impactStormsStr = " @ ".join(repr(storm) for storm in noise.impactStorms)
            if noise.actualNoiseFlag != noise.expectNoiseFlag:
                noise.passed = False
                # print(noise)
                # print(noise.impactStormsStr)
                # return
            self._noises.append(noise)

    def exportToCsv(self, path: str):
        CSVHandler.exportNoise(self._rawNoises, self._noises, path)