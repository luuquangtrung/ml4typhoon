# IMPORT 3rd PARTY MODULES
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import copy
import xesmf as xe

#### IMPORT NATIVE MODULES ####
from .utilities import *


class WeatherDataset:
    def __init__(self, min_lat: float = 0, max_lat: float = 0, step_lat: float = 0, dim_lat: int = 0, min_lon: float = 0, max_lon: float = 0, step_lon: float = 0, dim_lon: int = 0, step_time_hours: float=0):
        #### COMMON CONSTANTS ####
        self.MIN_LAT = min_lat
        self.MAX_LAT = max_lat
        self.STEP_LAT = step_lat
        self.MIN_LON = min_lon
        self.MAX_LON = max_lon
        self.STEP_LON = step_lon
        self.PREC_LAT = GetPrecision(self.STEP_LAT)
        self.PREC_LON = GetPrecision(self.STEP_LON)
        self.DIM_LAT = dim_lat
        self.DIM_LON = dim_lon
        self.STEP_TIME_HOURS = step_time_hours

    # This function is used to load raw data files from data provider
    # and perform some basic data filter operations
    # after that merge to one single xarray.dataset

    def LoadRaw(self, paths: list[str]) -> xr.Dataset:
        pass

    # This function is used to load saved data that have been processed earlier

    def LoadFromDisk(self, path: str) -> xr.Dataset:
        dataset = xr.load_dataset(filename_or_obj=path, engine="netcdf4")
        return dataset
    
    # This function is used to save the processed dataset
    # return 0 if success, otherwise fail

    def SaveToDisk(self, path: str, dataset: xr.Dataset) -> int:
        try:
            dataset.to_netcdf(path=path)
            return 0
        except Exception as e:
            return e

    def SelectData(
        self,
        dataset: xr.Dataset,
        lat_range: tuple[float, float], lon_range: tuple[float, float],
        datetime_range: list[tuple[datetime.date, datetime.time]]
    ) -> xr.Dataset:
        ds = dataset
        # Crop data by latitude and longitude
        ds = ds.where(ds.latitude <= lat_range[1], drop=True)
        ds = ds.where(ds.latitude >= lat_range[0], drop=True)
        ds = ds.where(ds.longitude <= lon_range[1], drop=True)
        ds = ds.where(ds.longitude >= lon_range[0], drop=True)
        # Crop data by time
        ds_sel = ds.where(
            ds.time.dt.date.isin([dt[0] for dt in datetime_range]) &
            ds.time.dt.time.isin([dt[1] for dt in datetime_range]),
            drop=True
        )
        # Check if selected data is empty or not
        if (all(not len(ds_sel[data_var].values) for data_var in list(ds_sel.keys()))):
            return None
        return ds_sel
    
    # This function is used to batch load the datasets that is saved using SaveToDisk
    # return a single dataset

    def LoadBatchFromDisk(self, paths: list[str]) -> xr.Dataset:
        ds_list = [
            xr.load_dataset(filename_or_obj=f, engine="netcdf4")
            for f in paths
        ]
        ds = xr.combine_nested(ds_list, concat_dim="time")
        return ds
    
    # This function is used to regrid data
    # return re-gridded dataset
    """
    Available methods
    - ‘bilinear’
    - ‘conservative’, need grid corner information
    - ‘conservative_normed’, need grid corner information
    - ‘patch’
    - ‘nearest_s2d’
    - ‘nearest_d2s’
    """

    def ReGrid(self, ds: xr.Dataset, lat_step: float, lon_step: float, method="conservative") -> xr.Dataset:
        # Build new grids from old grids and parameters
        old_lat_grid = np.array(ds["latitude"].values)
        old_lon_grid = ds["longitude"].values
        z_names = None
        keys = list(ds.indexes.keys())
        keys.remove("latitude")
        keys.remove("longitude")
        if not len(keys):
            z_names = None
        else:
            z_names = keys
        z_grid_list = [{str(z): ds[z].values} for z in z_names]
        lat_min = np.min(old_lat_grid)
        lat_max = np.max(old_lat_grid)
        lon_min = np.min(old_lon_grid)
        lon_max = np.max(old_lon_grid)
        lat_grid = np.arange(lat_min, lat_max, lat_step)
        lon_grid = np.arange(lon_min, lon_max, lon_step)
        # Build new container
        new_ds = xr.Dataset()
        new_ds = new_ds.assign_coords({"latitude": lat_grid})
        new_ds = new_ds.assign_coords({"longitude": lon_grid})
        for z_grid in z_grid_list:
            new_ds = new_ds.assign_coords(z_grid)
        # Build regridder
        regridder = xe.Regridder(ds, new_ds, method)
        # Regrid
        new_ds = regridder(ds)
        # Add attributes back to new dataset
        new_ds.attrs.update(ds.attrs)
        new_ds.attrs["REGRID_LAT"] = f"center={findMiddle(lat_grid.tolist())} min={np.min(lat_grid)} max={np.max(lat_grid)}"
        new_ds.attrs["REGRID_LON"] = f"center={findMiddle(lon_grid.tolist())} min={np.min(lon_grid)} max={np.max(lon_grid)}"
        return new_ds
    
    # GET SAMPLE

    def GetSample(self, wds: xr.Dataset, id: str, lat_c: float, lon_c: float, date_c: datetime.date, time_c: datetime.time, lat_dim: int = 0, lon_dim: int = 0, negative_type: str = None):
        lat_dim = lat_dim if lat_dim else self.DIM_LAT
        lon_dim = lon_dim if lon_dim else self.DIM_LON
        # LAT CALC
        lat_c_o = lat_c
        lat_grid = np.asarray(wds["latitude"].values)
        lat_step = self.STEP_LAT
        lat_c = FindNearest(array=lat_grid, value=lat_c)
        # LON CALC
        lon_c_o = lon_c
        lon_grid = np.asarray(wds["longitude"].values)
        lon_step = self.STEP_LON
        lon_c = FindNearest(array=lon_grid, value=lon_c)
        dt_r = [(date_c, time_c)]
        lat_dim = int(lat_dim/2)
        lon_dim = int(lon_dim/2)
        lat_r = (lat_c-lat_dim*lat_step, lat_c+lat_dim*lat_step)
        lon_r = (lon_c-lon_dim*lon_step, lon_c+lon_dim*lon_step)
        # Extract dataset
        s_wds = self.SelectData(wds, lat_r, lon_r, dt_r)
        if not s_wds:
            return None
        # Add metadata to dataset
        s_wds.attrs["SID"] = id
        s_wds.attrs["TYPE"] = "POSITIVE" if not negative_type else negative_type
        s_wds.attrs["ISO_TIME"] = f"{date_c.strftime('%Y-%m-%d')} {time_c.strftime('%H:%M:%S')}"
        s_wds.attrs["LAT"] = f"original={lat_c_o} center={lat_c} min={min(s_wds['latitude'].values)} max={max(s_wds['latitude'].values)}"
        s_wds.attrs["LON"] = f"original={lon_c_o} center={lon_c} min={min(s_wds['longitude'].values)} max={max(s_wds['longitude'].values)}"
        return s_wds
    
    # SAMPLE

    def Debug(self):
        print(self.MIN_LAT, self.MAX_LAT, self.STEP_LAT, self.PREC_LAT, self.MIN_LON, self.MAX_LON, self.STEP_LON, self.PREC_LON)
