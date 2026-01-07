import os
import multiprocessing as mp
import pandas as pd
import xarray as xr
import datetime

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from internals import MERRA2_IBTRACS, FNL_IBTRACS, MultiProcessing

def FixedDomain(WD:WeatherDataset, HT:HurricaneTrack, htdf:pd.DataFrame, first_htdf:pd.DataFrame, wd_path:str, output_path:str):
    filename = os.path.basename(wd_path)
    wds = WD.LoadFromDisk(wd_path)
    ds_isotime_str = "_".join(filename.split(".")[0].split("_")[1:])
    ds_isotime = datetime.datetime.strptime(ds_isotime_str, "%Y%m%d_%H_%M")
    _domain_loc = configs.FixedDomain.DOMAIN_COORD
    _lat = FindNearest(wds["latitude"].values, _domain_loc[0])
    _lon = FindNearest(wds["longitude"].values, _domain_loc[1])
    domain_loc = (_lat, _lon)
    domain_lat_max = domain_loc[0] + int(WD.DIM_LAT/2)*(WD.STEP_LAT)
    domain_lat_min = domain_loc[0] - int(WD.DIM_LAT/2)*(WD.STEP_LAT)
    domain_lon_max = domain_loc[1] + int(WD.DIM_LON/2)*(WD.STEP_LON)
    domain_lon_min = domain_loc[1] - int(WD.DIM_LON/2)*(WD.STEP_LON)
    TYPE = "NEGATIVE"
    ds_sid = None
    for i in htdf.index:
        id = htdf["SID"][i]
        lat_c = htdf["LAT"][i]
        lon_c = htdf["LON"][i]
        isotime_c = htdf["ISO_TIME"][i].to_pydatetime()
        if not (ds_isotime == isotime_c):
            continue
        if (lat_c <= domain_lat_max and lat_c >= domain_lat_min and lon_c <= domain_lon_max and lon_c >= domain_lon_min):
            lat_d = abs(lat_c - domain_loc[0])
            lon_d = abs(lon_c - domain_loc[1])
            TYPE = "NONE"
            if (lat_d <= configs.FixedDomain.CENTER_THRESHOLD and lon_d <= configs.FixedDomain.CENTER_THRESHOLD):
                if (((first_htdf["SID"] == id) & (first_htdf["ISO_TIME"] == ds_isotime)).any()):
                    TYPE = "POSITIVE"
                    ds_sid = id
            break
        pass
    if (TYPE == "NONE"):
        return
    s_wds = None
    save_path = None
    if (TYPE == "POSITIVE"):
        s_wds = WD.GetSample(wds, ds_sid, domain_loc[0], domain_loc[1], date_c=ds_isotime.date(), time_c=ds_isotime.time(), lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON)
        save_path = os.path.join(output_path, f"POSITIVE_{ds_sid}.nc")
    else:
        s_wds = WD.GetSample(wds, "None", domain_loc[0], domain_loc[1], date_c=ds_isotime.date(), time_c=ds_isotime.time(), lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON, negative_type="FIXED_DOMAIN")
        save_path = os.path.join(output_path, f"NEGATIVE_{filename}")
    WD.SaveToDisk(save_path, s_wds)
    return

def Worker(queue:mp.Queue, WD:WeatherDataset, HT:HurricaneTrack, htdf:pd.DataFrame, first_htdf:pd.DataFrame, output_path:str):
    while (queue.qsize()):
        wds_file = queue.get()
        if not wds_file:
            break
        FixedDomain(WD, HT, htdf, first_htdf, wds_file, output_path)
        pass
    print("Done.")
    exit()

def FixedDomain_Main(WD: WeatherDataset, HT: HurricaneTrack, PrepDir:str, OutDir:str, nworker:int=1):
    first_htdf_files = utilities.RecurseListDir(PrepDir, ["FIRST_*.csv"])
    htdf_files = utilities.RecurseListDir(PrepDir, ["FULL_*.csv"])
    wds_files = utilities.RecurseListDir(PrepDir, ["*.nc"])
    first_htdf = HT.LoadBatch(first_htdf_files)
    htdf = HT.LoadBatch(htdf_files)
    queue = mp.Queue()
    for f in wds_files:
        queue.put(f)
        pass
    MultiProcessing(Worker, (queue, WD, HT, htdf, first_htdf, OutDir), nworker)
    return

def Fnl():
    print("NCEP-FNL dataset extraction")
    FNL, IBTRACS = FNL_IBTRACS()
    out_dir = os.path.join(configs.paths.FNL_IBTRACS_OUT, "FixedDomain")
    utilities.CleanDir(out_dir)
    FixedDomain_Main(FNL, IBTRACS, configs.paths.FNL_IBTRACS_PREP, out_dir, configs.nworkers.FNL_FIXEDDOMAIN)
    print("NCEP-FNL dataset extraction: Done.")
    return

def Merra2():
    print("NASA-MERRA2 dataset extraction")
    MERRA2, IBTRACS = MERRA2_IBTRACS()
    out_dir = os.path.join(configs.paths.MERRA2_IBTRACS_OUT, "FixedDomain")
    utilities.CleanDir(out_dir)
    FixedDomain_Main(MERRA2, IBTRACS, configs.paths.MERRA2_IBTRACS_PREP, out_dir, configs.nworkers.MERRA2_FIXEDDOMAIN)
    print("NASA-MERRA2 dataset extraction: Done.")
    return

def Main():
    print("Fixed Domain")
    Fnl()
    Merra2()
    print("Fixed Domain: Done.")
    return

if __name__=="__main__":
    Main()