# This file extends __prototype__.py and written to process ncep_fnl raw data
from .__prototype__ import *


class NcepFnl(WeatherDataset):
    def __init__(self, min_lat: float = -50, max_lat: float = 70, step_lat: float = 1, dim_lat: int = 17, min_lon: float = 60, max_lon: float = 220, step_lon: float = 1, dim_lon: int = 17, step_time_hours: float = 6.0):
        super().__init__(min_lat, max_lat, step_lat, dim_lat, min_lon, max_lon, step_lon, dim_lon, step_time_hours)
        #### CUSTOM CONSTANTS ####
        self.EXTRACTION_INFO = [
            {
                "typeOfLevel": "isobaricInhPa",
                "cfVarName_list": ["u", "v", "w", "absv", "t", "gh", "r"],
                "renameVars": {"u": "ugrdprs", "v": "vgrdprs", "w": "vvelprs", "t": "tmpprs", "gh": "hgtprs", "r": "rhprs"}
            },
            {
                "typeOfLevel": "surface",
                "cfVarName_list": ["t", "sp", "lsm"],
                "renameVars": {"t": "tmpsfc", "sp": "pressfc", "lsm": "landmask"}
            },
            {
                "typeOfLevel": "tropopause",
                "cfVarName_list": ["gh", "t"],
                "renameVars": {"gh": "hgttrp", "t": "tmptrp"}
            },
        ]
        self.COORDS_NAME = ["latitude", "longitude", "time", "isobaricInhPa"]

    # This function is used to load raw data files from data provider
    # and perform some basic data filter operations
    # after that merge to one single xarray.dataset

    def LoadRaw(self, paths: list[str]) -> xr.Dataset:
        ds_list = []
        for ex_i in self.EXTRACTION_INFO:
            ds_l = []
            for varname in ex_i["cfVarName_list"]:
                ds_raw = [
                    xr.open_dataset(f, engine="cfgrib", backend_kwargs={
                        "filter_by_keys": {
                            "cfVarName": varname, "typeOfLevel": ex_i["typeOfLevel"]
                        },
                        "indexpath": ""
                    })
                    for f in paths
                ]
                ds_raw = [
                    d.drop([v for v in (list(d.coords) + list(d.data_vars))
                           if v not in (list(d.indexes) + list(d.keys()) + ["time", "step"])])
                    for d in ds_raw
                ]
                ds_l.append(xr.combine_nested(ds_raw, concat_dim="time"))
            ds_tmp = xr.merge(ds_l)
            if ex_i["renameVars"]:
                ds_tmp = ds_tmp.rename(ex_i["renameVars"])
            ds_list.append(ds_tmp)
        ds = xr.merge(ds_list, compat="override")
        return ds

    # This function is used to process the loadded raw dataset

    def ProcessRaw(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = dataset
        ds = ds.drop_vars("step")
        # Crop to region of interest
        ds = ds.where(ds.latitude <= self.MAX_LAT, drop=True)
        ds = ds.where(ds.latitude >= self.MIN_LAT, drop=True)
        ds = ds.where(ds.longitude <= self.MAX_LON, drop=True)
        ds = ds.where(ds.longitude >= self.MIN_LON, drop=True)
        ds = ds.sortby("isobaricInhPa", ascending=False)
        return ds

    def FixLevel(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = dataset
        for c in self.COORDS_NAME:
            ds = ds.set_coords(c)
        ds = ds.reset_index("isobaricInhPa")
