import os
import torch

import pandas as pd
import numpy as np
import xarray as xr

from torch.utils.data import Dataset
from pathlib import Path

from scipy.ndimage import zoom

import datetime
import glob



#################
# General Dataset
#################
# hàm đọc file input dựa trên data_type (merra hay fnl)
def read_data(path, data_type):
    if data_type == 0: # FNL
        SINGLE_VAR = ['tmpsfc', 'pressfc', 'landmask', 'hgttrp', 'tmptrp']
        PRESS_VAR = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'hgtprs', 'rhprs']
        LEVEL = 21

        image = []
        ds = xr.open_dataset(path)
        for var in SINGLE_VAR:
            tmp = ds.variables[var].data[0]
            image.append(tmp)
        for var in PRESS_VAR:
            tmp =ds.variables[var].data[0][:LEVEL]
            image.extend(tmp)

        image = np.array(image)

        return image


    else: #merra
        SINGLE_VAR = ['PHIS','PS','SLP']
        PRESS_VAR = ['EPV','H','O3','OMEGA','QI','QL','QV','RH','T','U','V']
        LEVEL = 21


        image = []
        ds = xr.open_dataset(path)
        for var in SINGLE_VAR:
            tmp = ds.variables[var].data
            tmp = np.nan_to_num(tmp)
            tmp = resize_image(tmp)
            image.append(tmp)
        for var in MULTI_VAR:
            tmp =ds.variables[var].data[:LEVEL]
            tmp = np.nan_to_num(tmp)
            tmp = resize_image(tmp)
            image.extend(tmp)
        image = np.array(image)
        return image


def aggregate_func(list_data):
    alpha = 0.85
    agg_ft = np.zeros(list_data[0].shape, dtype=np.float32)
        
    for step in range(len(list_data)):

        step_ft = list_data[step]
        
        agg_ft = agg_ft + np.multiply(step_ft, alpha ** step)
            
    agg_ft = torch.tensor(np.array(agg_ft), dtype=torch.float)

    return agg_ft


class BaseDataset(Dataset):
    def __init__(self,
                 csv_path,
                 data_type,
                 run_opt,
                 agg_steps,
    ):
        self.df = pd.read_csv(csv_path)

        # group theo từng thời gian, vị trí
        # nếu aggregate, mỗi group con sẽ có đường dẫn file của nhiều step 0, -1, -2, ...
        # nếu không aggregate, mỗi group con chỉ có đường dẫn file của step 0
        self.groups = list(self.df.groupby(['datetime', 'point'], as_index=False))

        self.data_type = data_type
        self.run_opt = run_opt
        self.agg_steps = agg_steps

    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):

        (date_time, point), df = self.groups[idx]

        if self.run_opt == 1: # aggregate

            list_data = []
            for i in range(self.agg_steps):
                # lấy dường dẫn
                data_path = df.loc[df['step'] == i * (-1),  'path'].values[0]
                # đọc file, tùy theo loại data (fnl, merra)
                data = read_data(data_path, self.data_type)

                list_data.append(data)

            # tính aggregate
            result = aggregate_func(list_data)

        elif self.run_opt == 0: # no aggregate
            # lấy dường dẫn
            data_path = df['path'].values[0]

            # đọc file, tùy theo loại data (fnl, merra)
            result = read_data(data_path, self.data_type)


        # normalize, chưa có code, tùy mọi người sửa

        # trả về thông tin về thời gian, point(lat, lon), tensor
        return date_time, point, result