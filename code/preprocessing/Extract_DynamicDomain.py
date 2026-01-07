import os
import multiprocessing as mp
import pandas as pd
import xarray as xr
import datetime
import math

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from internals import MERRA2_IBTRACS, FNL_IBTRACS, MultiProcessing

def DynamicDomain(WD: WeatherDataset, HT: HurricaneTrack, htdf:pd.DataFrame, wd_files:list[str], output_path:str, ndomains:list[str], n_steps:int):
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
        lat_c = findMiddle(s_ds["latitude"].values.tolist())
        lon_c = findMiddle(s_ds["longitude"].values.tolist())
        current_datetime = htdf["ISO_TIME"][i]
        t = 0
        while True:
            # Negative extraction
            for domain in ndomains:
                match domain:
                    case "n":
                        n_lat_c = lat_c + int(WD.DIM_LAT)*WD.STEP_LAT
                        n_lon_c = lon_c + 0
                    case "ne":
                        n_lat_c = lat_c + int(WD.DIM_LAT)*WD.STEP_LAT
                        n_lon_c = lon_c + int(WD.DIM_LON)*WD.STEP_LON
                    case "e":
                        n_lat_c = lat_c + 0
                        n_lon_c = lon_c + int(WD.DIM_LON)*WD.STEP_LON
                    case "se":
                        n_lat_c = lat_c - int(WD.DIM_LAT)*WD.STEP_LAT
                        n_lon_c = lon_c + int(WD.DIM_LON)*WD.STEP_LON
                    case "s":
                        n_lat_c = lat_c - int(WD.DIM_LAT)*WD.STEP_LAT
                        n_lon_c = lon_c + 0
                    case "sw":
                        n_lat_c = lat_c - int(WD.DIM_LAT)*WD.STEP_LAT
                        n_lon_c = lon_c - int(WD.DIM_LON)*WD.STEP_LON
                    case "w":
                        n_lat_c = lat_c - 0
                        n_lon_c = lon_c - int(WD.DIM_LON)*WD.STEP_LON
                    case "nw":
                        n_lat_c = lat_c + int(WD.DIM_LAT)*WD.STEP_LAT
                        n_lon_c = lon_c - int(WD.DIM_LON)*WD.STEP_LON
                    case _:
                        raise f"{domain} is invalid!!!"
                n_s_ds = WD.GetSample(w_ds, id, n_lat_c, n_lon_c, date_c, time_c, lat_dim=WD.DIM_LAT, lon_dim=WD.DIM_LON, negative_type=f"DYNAMIC_{domain}_{t}")
                if not n_s_ds:
                    continue
                save_path = os.path.join(output_path, f"NEGATIVE_{id}_{domain}_{t}.nc")
                WD.SaveToDisk(save_path, n_s_ds)
                pass
            # Return to the past
            t += 1
            if t > n_steps:
                break
            selected_datetime = current_datetime - datetime.timedelta(hours=(WD.STEP_TIME_HOURS*(t)))
            date_c = selected_datetime.date()
            time_c = selected_datetime.time()
            target = f"{date_c.strftime('%Y%m%d')}_{time_c.strftime('%H')}_{time_c.strftime('%M')}"
            target_path = [path for path in wd_files if str(os.path.basename(path)).__contains__(target)]
            if len(target_path) != 1:
                continue
            target_path = target_path[0]
            w_ds = WD.LoadFromDisk(target_path)
            pass
        pass
    return

def Worker(queue:mp.Queue, WD:WeatherDataset, HT:HurricaneTrack, wd_files:list[str], output_path:str, ndomains:list[str], nsteps:int):
    while (queue.qsize()):
        htdf = queue.get()
        if (not type(htdf) == pd.DataFrame):
            continue
        DynamicDomain(WD, HT, htdf, wd_files, output_path, ndomains, nsteps)
        pass
    print("Done.")
    exit()

def DynamicDomain_Main(WD: WeatherDataset, HT: HurricaneTrack, PrepDir:str, OutDir:str, nworker:int=1):
    ht_files = utilities.RecurseListDir(PrepDir, ["FIRST_*.csv"])
    wd_files = utilities.RecurseListDir(PrepDir, ["*.nc"])
    htdf = HT.LoadBatch(ht_files)
    htdfs = HT.Split(htdf, configs.nworkers.BATCH_SIZE)
    queue = mp.Queue()
    for h in htdfs:
        queue.put(h)
        pass
    MultiProcessing(Worker, (queue, WD, HT, wd_files, OutDir, configs.DynamicDomain.DOMAIN_LOCATIONS, configs.DynamicDomain.N_STEPS), nworker)
    return

def Fnl():
    print("NCEP-FNL dataset extraction")
    FNL, IBTRACS = FNL_IBTRACS()
    out_dir = os.path.join(configs.paths.FNL_IBTRACS_OUT, "DynamicDomain")
    utilities.CleanDir(out_dir)
    DynamicDomain_Main(FNL, IBTRACS, configs.paths.FNL_IBTRACS_PREP, out_dir, configs.nworkers.FNL_DYNAMICDOMAIN)  
    print("NCEP-FNL dataset extraction: Done.")
    return

def Merra2():
    print("NASA-MERRA2 dataset extraction")
    MERRA2, IBTRACS = MERRA2_IBTRACS()
    out_dir = os.path.join(configs.paths.MERRA2_IBTRACS_OUT, "DynamicDomain")
    utilities.CleanDir(out_dir)
    DynamicDomain_Main(MERRA2, IBTRACS, configs.paths.MERRA2_IBTRACS_PREP, out_dir, configs.nworkers.MERRA2_DYNAMICDOMAIN)      
    print("NASA-MERRA2 dataset extraction: Done")
    return

def Main():
    print("Dynamic Domain")
    Fnl()
    Merra2()
    print("Dynamic Domain: Done.")
    return

if __name__=="__main__":
    Main()