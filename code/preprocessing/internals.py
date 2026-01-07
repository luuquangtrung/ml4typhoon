import configparser
import multiprocessing as mp
from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *

def MultiProcessing(Worker, args:tuple, n_worker:int):
    print(f"MultiProcess: {n_worker}")
    ps = []
    for i in range(n_worker):
        p = mp.Process(
            target=Worker,
            args=args
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def FNL_IBTRACS():
    FNL = NcepFnl()
    IBTRACS = Ibtracs()
    IBTRACS.TIME_POINTS_HOURS = [0, 6, 12, 18]
    return FNL, IBTRACS

def MERRA2_IBTRACS():
    MERRA2 = NasaMerra2()
    IBTRACS = Ibtracs()
    IBTRACS.TIME_POINTS_HOURS = [0,3,6,9,12,15,18,21]
    return MERRA2, IBTRACS
