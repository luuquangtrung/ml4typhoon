# This file extends nasa_merra2.py and written to process wrf_cmip5 raw data
import xarray as xr
"""""
from nasa_merra2 import NasaMerra2


class WrfCmip5(NasaMerra2):
    def __init__(self, min_lat: float = -50, max_lat: float = 70, step_lat: float = 0.5, dim_lat: int = 33, min_lon: float = 60, max_lon: float = 220, step_lon: float = 0.625, dim_lon: int = 33, step_time_hours:float = 3.0):
        super().__init__(min_lat, max_lat, step_lat, dim_lat, min_lon, max_lon, step_lon, dim_lon, step_time_hours)
        #### CUSTOM CONSTANTS ####
        self.RENAME_VARS = {
            "lat": "latitude",
            "lon": "longitude",
            "lev": "isobaricInhPa"
        }
"""
if __name__=="__main__":
    data = xr.load_dataset(filename_or_obj="/N/u/tqluu/BigRed200/workspace/libtcg_Dataset2/data/raw/wrf_cmip5/wrf_cmip5/baseline_18km/raw_wrfout_d01_2005-12-31_00:00:00", engine="netcdf4")
    #print(data)
    #print(data["XLAT"])
    print(data["XLONG"])


    