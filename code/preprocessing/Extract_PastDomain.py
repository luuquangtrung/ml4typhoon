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

def PastDomain(WD: WeatherDataset, HT: HurricaneTrack, htdf:pd.DataFrame, wd_files:list[str], output_path:str, nstep:int):
    for i in htdf.index:
        # Find for postitive sample
        id = htdf["SID"][i]
        lat_c = htdf["LAT"][i]
        lon_c = htdf["LON"][i]
        date_c = htdf["ISO_TIME"][i].date()
        time_c = htdf["ISO_TIME"][i].time()
        target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
        target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
        if len(target_path) != 1:
            continue
        target_path = target_path[0]
        w_ds = WD.LoadFromDisk(target_path)
        lat_c = FindNearest(w_ds["latitude"].values, lat_c)
        lon_c = FindNearest(w_ds["longitude"].values, lon_c)
        s_ds = WD.GetSample(w_ds, id, lat_c, lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON)
        if not s_ds:
            continue
        save_path = os.path.join(output_path, f"POSITIVE_{id}.nc")
        WD.SaveToDisk(save_path, s_ds)
        current_datetime = htdf["ISO_TIME"][i]
        for j in range(nstep):
            selected_datetime = current_datetime - datetime.timedelta(hours=(WD.STEP_TIME_HOURS*(j+1)))
            date_c = selected_datetime.date()
            time_c = selected_datetime.time()
            target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
            target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
            if len(target_path) != 1:
                continue
            target_path = target_path[0]
            n_w_ds = WD.LoadFromDisk(target_path)
            n_s_ds = WD.GetSample(n_w_ds, id, lat_c, lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON, negative_type=f"PAST_T-{j+1}")
            if not n_s_ds:
                continue
            datetime.datetime.strftime
            save_path = os.path.join(output_path, f"NEGATIVE_{id}_{j+1}_{selected_datetime.strftime('%Y%m%d_%H%M')}.nc")
            WD.SaveToDisk(save_path, n_s_ds)
            pass
        pass
    return

def Worker(queue:mp.Queue, WD:WeatherDataset, HT:HurricaneTrack, wd_files:list[str], output_path:str, nstep:int):
    while (queue.qsize()):
        htdf = queue.get()
        if (not type(htdf) == pd.DataFrame):
            continue
        PastDomain(WD, HT, htdf, wd_files, output_path, nstep)
    print("Done.")
    exit()

def PastDomain_Main(WD: WeatherDataset, HT: HurricaneTrack, PrepDir:str, OutDir:str, nworker:int=1):
    ht_files = utilities.RecurseListDir(PrepDir, ["FIRST_*.csv"])
    wd_files = utilities.RecurseListDir(PrepDir, ["*.nc"])
    htdf = HT.LoadBatch(ht_files)
    htdfs = HT.Split(htdf, configs.nworkers.BATCH_SIZE)
    queue = mp.Queue()
    for h in htdfs:
        queue.put(h)
    MultiProcessing(Worker, (queue, WD, HT, wd_files, OutDir, configs.PastDomain.STEP_BACK_COUNT), nworker)
    return

def Fnl():
    print("NCEP-FNL dataset extraction")
    FNL, IBTRACS = FNL_IBTRACS()
    out_dir = os.path.join(configs.paths.FNL_IBTRACS_OUT, "PastDomain")
    utilities.CleanDir(out_dir)
    PastDomain_Main(FNL, IBTRACS, configs.paths.FNL_IBTRACS_PREP, out_dir, configs.nworkers.FNL_PASTDOMAIN)   
    print("NCEP-FNL dataset extraction: Done.")
    return

def Merra2():
    print("NASA-MERRA2 dataset extraction")
    MERRA2, IBTRACS = MERRA2_IBTRACS()
    out_dir = os.path.join(configs.paths.MERRA2_IBTRACS_OUT, "PastDomain")
    utilities.CleanDir(out_dir)
    PastDomain_Main(MERRA2, IBTRACS, configs.paths.MERRA2_IBTRACS_PREP, out_dir, configs.nworkers.MERRA2_PASTDOMAIN)   
    print("NASA-MERRA2 dataset extraction: Done.")
    return

def Main():
    print("Past Domain")
    Fnl()
    Merra2()
    print("Past Domain: Done.")
    return

if __name__=="__main__":
    Main()