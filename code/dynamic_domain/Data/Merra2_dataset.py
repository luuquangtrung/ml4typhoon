import sys
sys.path.insert(0, 'path-to-Lib')

import os
import torch
import json
import cv2

import numpy as np
import xarray as xr
import pandas as pd

from time import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2

from pandarallel import pandarallel
from sys import getsizeof

from Utils.New_features import *



# tqdm.pandas()


SINGLE_VAR = ['PHIS', 'PS', 'SLP',]
PRESS_VAR = ['H', 'O3', 'OMEGA', 'QI', 'QL', 'QV', 'RH', 'T', 'U', 'V']

SINGLE_VAR = ['PHIS', 'PS', 'SLP',]
PRESS_VAR = ['H', 'OMEGA', 'QI', 'QL', 'QV', 'RH', 'T', 'U', 'V']

# SINGLE_VAR = ['PHIS', 'PS',]
# PRESS_VAR = ['QI', 'QL', 'QV', 'U', 'V']

LEVEL = 25

PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825, 
               800, 775, 750, 725, 700, 650, 600, 550, 
               500, 450, 400, 350, 300, 250, 200, 150, 100]

# number of channels after concat
INP_CHANNELS = len(SINGLE_VAR) + LEVEL * len(PRESS_VAR)

LIST_VAR = [var + '0' for var in SINGLE_VAR]
LIST_VAR.extend([var + str(level) for var in PRESS_VAR for level in PRESS_LEVEL])

#===============================
# Description: 
#   - Fully loaded dataset before train progress
#   - Apply standard normalization to dataset
#   - Apply aggregate (OPTIONAL)
#===============================
class Merra2_dataset(Dataset):
    def __init__(self, 
                 merra_path: Path,
                 stat_path: Path = 'path-to-statistics.xlsx',
                 dtype: str = 'clf',
                 agg_step: int = 0,
                 agg_alpha: float = 0.85,
                 num_workers: int = 1,
                 transform = None,
                 pre_load: bool = False,
                ):
        
        self.data_path = pd.read_csv(merra_path)#.iloc[:100]
        self.stat = pd.read_excel(stat_path)
        self.stat['variable_name'] = self.stat['variable'] + self.stat['level'].astype(str)
        self.stat.set_index('variable_name', inplace=True)
        self.mean = self.stat['mean'].loc[LIST_VAR].to_numpy()[:, np.newaxis, np.newaxis]
        self.std = self.stat['std'].loc[LIST_VAR].to_numpy()[:, np.newaxis, np.newaxis]
        self.max = self.stat['max'].loc[LIST_VAR].to_numpy()
        self.dtype = dtype
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.num_workers = num_workers
        self.pre_load = pre_load
        self.transform = transform
        self.read_data = self.read_data
        self.data_path = self.data_path[~ self.data_path['Label'].isna()].reset_index(drop=True)
        if pre_load:
            self.load_data()
            

    def read_data(self, row):
        nc_path = row['Path']
        input = []
        ds = xr.open_dataset(nc_path)
        for var in SINGLE_VAR:
            arr = ds.variables[var].data
            input.append(arr)

        for var in PRESS_VAR:
            arr = ds.variables[var].data[: LEVEL]
            input.extend(arr)
        ds.close()
        res = (np.array(input) - self.mean) / self.std # norm
        res[np.isnan(res)] = 0
        
        if self.transform is not None:
            res = self.transform(res)

        return res
        
    # Load data and agg into a sample
    def get_data(self, sample):
        id, position, step = sample[['ID', 'Position', 'Step']]
        
        # filter sample within agg range
        filter = self.data_path[
            (self.data_path['ID'] == id) &
            (self.data_path['Position'] == position) &
            (self.data_path['Step'].between(step, step + self.agg_step, inclusive='both'))
        ].copy()
        filter['Weight'] = self.agg_alpha ** (filter['Step'] - step) # calculate weight for each step
        
        return filter.apply(self.read_data, axis=1).sum()
    
    # resize image
    def transform_input(self, x):
        """
        x: V, H, W
        """
        x = np.transpose(x, axes=(1, 2, 0)) # H, W, V
        x = cv2.resize(x, (32, 32))
        x = np.transpose(x, axes=(2, 0, 1)) # V, H, W
        return x

    # Fully load data when pre_load = True
    def load_data(self):
        print("Loading...")
        timer = time()
        pandarallel.initialize(nb_workers=self.num_workers)
        self.data_arr = np.array(self.data_path[~ self.data_path['Label'].isna()].parallel_apply(self.get_data, axis=1).to_list())
        
        print('[INFO]: Dataset loaded! Size:', len(self.data_path))
        print('[INFO]: Dataset loaded! Time:', f'{(time() - timer):.2f}')
        print('[INFO]: Dataset include', (self.data_path['Label'] == 1).sum(), 'POS and', (self.data_path['Label'] == 0).sum(), 'NEG')
        print('[INFO]: Dataset size on MEM:', getsizeof(self.data_arr))
        
    def __len__(self):
        return len(self.data_path[~ self.data_path['Label'].isna()])
    
    def __getitem__(self, idx):
        # print(idx)
        if self.pre_load:
            input = self.data_arr[idx]
        else:
            input = self.read_data(self.data_path.iloc[idx])
        
        input = torch.tensor(input, dtype=torch.float)
        
        label = self.data_path.iloc[idx]['Label']
        if self.dtype == 'clf':
            label = torch.tensor(label, dtype=torch.float)#.type(torch.LongTensor)
        
        else:
            label = torch.tensor(label, dtype=torch.float)
        # print(self.data_path.iloc[idx],label)
        # print(input.shape, input.min(), input.max(), input.mean())
        return input, label
    
    
class Merra2_expert_dataset(Merra2_dataset):
    def __init__(self, 
                 merra_path: Path,
                 stat_path: Path = 'path-to-statistics.xlsx',
                 dtype: str = 'clf',
                 agg_step: int = 0,
                 agg_alpha: float = 0.85,
                 num_workers: int = 1,
                 transform = False,
                 pre_load: bool = False,):
        
        self.T_VAR = ['RH', 'T', 'T', 'H', 'OMEGA', 'U', 'U', 'V', 'V', 'VOR', 'VOR', 'VOR', 'DIV']
        self.T_PRS = [750, 900, 500, 500, 500, 800, 200, 800, 200, 900, 700, 200, 200]
        self.T_CAT = ['RH750', 'T900', 'T500', 'H500', 'OMEGA500', 'U800', 'U200', 'V800', 'V200', 'VOR900', 'VOR700', 'VOR200', 'DIV200']
        self.T_IDX = [10, 4, 16, 16, 16, 8, 22, 8, 22, 4, 12, 22, 22]
        
        self.data_path = pd.read_csv(merra_path)
        self.stat = pd.read_excel(stat_path)
        self.stat['variable_name'] = self.stat['variable'] + self.stat['level'].astype(str)
        self.stat.set_index('variable_name', inplace=True)
        self.mean = self.stat['mean'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
        self.std = self.stat['std'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
        
        self.dtype = dtype
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.num_workers = num_workers
        self.pre_load = pre_load
        self.transform = transform
        if pre_load:
            self.load_data()
        else:
            self.data_path = self.data_path[~ self.data_path['Label'].isna()].reset_index(drop=True)
            
    # Load data and agg into a sample
    def get_data(self, sample):
        id, position, step = sample[['ID', 'Position', 'Step']]
        
        # filter sample within agg range
        filter = self.data_path[
            (self.data_path['ID'] == id) &
            (self.data_path['Position'] == position) &
            (self.data_path['Step'].between(step, step + self.agg_step, inclusive='both'))
        ].copy()
        filter['Weight'] = self.agg_alpha ** (filter['Step'] - step) # calculate weight for each step
        
        # Read data and apply norm
        def read_data(row):
            nc_path, weight = row[['Path', 'Weight']]
            input = []
            ds = xr.open_dataset(nc_path)
            
            for var, idx in zip(self.T_VAR[: - 4], self.T_IDX[: - 4]):
                arr = ds.variables[var].data[idx]
                input.append(arr)
                
            for idx in self.T_IDX[- 4: - 1]:
                U = ds.variables['U'].data[idx: idx + 1]
                V = ds.variables['V'].data[idx: idx + 1]
                lon = ds.coords['longitude'].data
                lat = ds.coords['latitude'].data[:: -1]
                
                lat_grid, lon_grid = meshgrid(lat, lon, 1)
                VOR = vorticity(U, V, lat_grid, lon_grid)
                input.extend(VOR)
                
            for idx in self.T_IDX[- 1:]:
                U = ds.variables['U'].data[idx: idx + 1]
                V = ds.variables['V'].data[idx: idx + 1]
                lon = ds.coords['longitude'].data
                lat = ds.coords['latitude'].data[:: -1]
                
                lat_grid, lon_grid = meshgrid(lat, lon, 1)
                DIV = divergence(U, V, lat_grid, lon_grid)
                input.extend(DIV)
                
            
            # res[np.isnan(res)] = np.take(self.max * 2, np.isnan(res).nonzero()[0])
            res = (np.array(input) - self.mean) / self.std * weight # norm
            res[np.isnan(res)] = 0
            
            if self.transform:
                res = self.transform_input(res)
            
            return res
        
        return filter.apply(read_data, axis=1).sum()
    
class Merra2_filter_dataset(Merra2_dataset):
    def __init__(self, 
                 merra_path: Path,
                 stat_path: Path = 'path-to-statistics.xlsx',
                 dtype: str = 'clf',
                 agg_step: int = 0,
                 corr:float = 0.3,
                 agg_alpha: float = 0.85,
                 num_workers: int = 1,
                 transform = False,
                 pre_load: bool = False,):
        
        PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825, 
               800, 775, 750, 725, 700, 650, 600, 550, 
               500, 450, 400, 350, 300, 250, 200, 150, 100]
        
        
        
        self.features = pd.read_excel("path-to-list-features-by-corr.xlsx")[["var",f"corr_{corr}"]]
        vars = self.features[self.features[f"corr_{corr}"] == 1]["var"].tolist()
        self.LIST_VAR = [i.split("_")[0] for i in vars]
        self.LIST_PR = [int(i.split("_")[1]) for i in vars]
        self.LIST_IDX = [PRESS_LEVEL.index(prs) if prs != 0 else -1 for prs in self.LIST_PR]
        self.T_CAT = [f'{self.LIST_VAR[i]}{self.LIST_PR[i]}' for i in range(len(self.LIST_VAR))]
        
        print("num of channel",len(self.LIST_VAR))
        
        
        self.data_path = pd.read_csv(merra_path)
        self.stat = pd.read_excel(stat_path)
        self.stat['variable_name'] = self.stat['variable'] + self.stat['level'].astype(str)
        self.stat.set_index('variable_name', inplace=True)
        self.mean = self.stat['mean'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
        self.std = self.stat['std'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
        print(self.mean.shape)
        self.dtype = dtype
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.num_workers = num_workers
        self.pre_load = pre_load
        self.transform = transform
        if pre_load:
            self.load_data()
        else:
            self.data_path = self.data_path[~ self.data_path['Label'].isna()].reset_index(drop=True)
    
    def get_data(self, sample):
        id, position, step = sample[['ID', 'Position', 'Step']]
        
        # filter sample within agg range
        filter = self.data_path[
            (self.data_path['ID'] == id) &
            (self.data_path['Position'] == position) &
            (self.data_path['Step'].between(step, step + self.agg_step, inclusive='both'))
        ].copy()
        filter['Weight'] = self.agg_alpha ** (filter['Step'] - step) # calculate weight for each step
        
        # Read data and apply norm
        def read_data(row):
            nc_path, weight = row[['Path', 'Weight']]
            input = []
            ds = xr.open_dataset(nc_path)
            
            for var, idx in zip(self.LIST_VAR, self.LIST_IDX):
                if idx == -1:
                    arr = ds.variables[var].data
                else:
                    arr = ds.variables[var].data[idx]
                input.append(arr)
            
            # res[np.isnan(res)] = np.take(self.max * 2, np.isnan(res).nonzero()[0])
            res = (np.array(input) - self.mean) / self.std * weight # norm
            res[np.isnan(res)] = 0
            
            if self.transform:
                res = self.transform_input(res)
            
            return res
        
        return filter.apply(read_data, axis=1).sum()

#===================
# Description: Define 3 DataLoader (Train, Val, Test)
#===================    
def prepare_Dataset(DatasetClass = Merra2_dataset,
                    train_path: Path = 'path',
                    val_path: Path = 'path',
                    test_path: Path = 'path',
                    batch_size: int = 32,
                    pin_memory: bool = torch.cuda.is_available(),
                    num_workers: int = os.cpu_count(),
                    out_dir: Path = None,
                    **kwargs):
    
    trainLoader = DataLoader(
        DatasetClass(merra_path=train_path, num_workers=num_workers, **kwargs),
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False
    )
    
    valLoader = DataLoader(
        DatasetClass(merra_path=val_path, num_workers=num_workers, **kwargs),
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False
    )
    test_kwargs = kwargs.copy()
    test_kwargs['pre_load'] = False
    testLoader = DataLoader(
        DatasetClass(merra_path=test_path, num_workers=num_workers, **test_kwargs),
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=False
    )

    return trainLoader, valLoader, testLoader