# This file extends __prototype__.py and written to process nasa_merra2 raw data
from .__prototype__ import *


class NasaMerra2(WeatherDataset):
    def __init__(self, min_lat: float = -50, max_lat: float = 70, step_lat: float = 0.5, dim_lat: int = 33, min_lon: float = 60, max_lon: float = 220, step_lon: float = 0.625, dim_lon: int = 33, step_time_hours:float = 3.0):
        super().__init__(min_lat, max_lat, step_lat, dim_lat, min_lon, max_lon, step_lon, dim_lon, step_time_hours)
        #### CUSTOM CONSTANTS ####
        self.RENAME_VARS = {
            "lat": "latitude",
            "lon": "longitude",
            "lev": "isobaricInhPa"
        }

    # This function is used to load raw data files from data provider
    # and perform some basic data filter operations
    # after that merge to one single xarray.dataset

    def LoadRaw(self, paths: list[str]) -> xr.Dataset:
        ds_list = [xr.load_dataset(
            filename_or_obj=f, engine="netcdf4") for f in paths]
        ds_tmp = xr.merge(ds_list)
        # Rename vars to syncs with other datasets
        ds = ds_tmp.rename(self.RENAME_VARS)
        return ds
    
    # This function is used to process the loadded raw dataset

    def ProcessRaw(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = dataset
        # Change axis of longitude
        lon_original = ds["longitude"]
        lon_normalized = [RoundBase(
            lon + 360, prec=3, base=self.STEP_LON) if lon < 0 else lon for lon in lon_original]
        ds["longitude"] = lon_normalized
        # Fix latitude
        lat_original = ds["latitude"]
        lat_normalized = [RoundBase(lat, prec=1, base=self.STEP_LAT)
                          for lat in lat_original]
        ds["latitude"] = lat_normalized
        # Crop to region of interest
        ds = ds.where(ds.latitude <= self.MAX_LAT, drop=True)
        ds = ds.where(ds.latitude >= self.MIN_LAT, drop=True)
        ds = ds.where(ds.longitude <= self.MAX_LON, drop=True)
        ds = ds.where(ds.longitude >= self.MIN_LON, drop=True)
        # Sort data values
        ds = ds.sortby("longitude")
        ds = ds.sortby("latitude")
        ds = ds.sortby("time")
        return ds
    
    # This function is used to split one large dataset
    # to smaller dataset. (one timestamp each dataset).

    def Split(self, dataset: xr.Dataset) -> list[xr.Dataset]:
        ds = dataset
        ds_list = []
        for t in list(ds.indexes["time"]):
            sub_ds = ds.sel(time=t)
            sub_ds.attrs["ISO_TIME"] = t.strftime("%Y-%m-%d %H:%M:%S")
            ds_list.append(copy.deepcopy(sub_ds))
        return ds_list
