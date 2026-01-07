from torch.utils.data import Dataset
import random
import pandas as pd
import numpy as np
import os
import torch
import xarray as xr

import matplotlib.pyplot as plt

def readMerra2(ncpath):
    SINGLE_VAR = ['PHIS','PS','SLP']
    PRESS_VAR = ["H", "OMEGA", "QI", "QL", "QV", "RH", "T", "U", "V"]
    LEVEL = 21


    image = []
    ds = xr.open_dataset(ncpath)
    for var in PRESS_VAR:
        tmp =ds.variables[var].data[:LEVEL]
        tmp = np.nan_to_num(tmp)
        image.extend(tmp)
    for var in SINGLE_VAR:
        tmp = ds.variables[var].data
        tmp = np.nan_to_num(tmp)
        image.append(tmp)
    image = np.array(image)
    return image




class MerraDataset(Dataset):
    def __init__(self, 
                 data, 
                 pos_ind, 
                 norm_type='new', 
                 small_set=False, 
                 preprocessed_dir="/N/scratch/tnn3/data_fullmap",
                 ):
        
        self.data = data
        self.pos_ind = pos_ind
        self.labels = self.data['Label']
        self.paths = self.data['Path'].values
        self.filenames = self.data['Filename'].values
        self.norm_type = norm_type
        self.small_set = small_set
        self.preprocessed_dir = preprocessed_dir

        if self.norm_type == 'new':

            self.stats_file = "csv/data_statistics.xlsx"
        else:
            self.stats_file = None
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Loading stacked data in .pt format
        # filename = self.filenames[idx].replace(".nc", ".pt")  # Convert .nc filename to .pt
        label = self.labels.iloc[idx]

        # file_path = os.path.join(self.preprocessed_dir, filename)

        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        # Load preprocessed tensor
        # data = torch.load(file_path).to(dtype=torch.float32)
        data = readMerra2(self.filenames[idx])


        # Normalised
        if self.stats_file:
            stat_df = pd.read_excel(self.stats_file)
                
            means = stat_df["Mean"].values.astype(np.float32)
            stds = stat_df["Std"].values.astype(np.float32)
            means = means.reshape(-1, 1, 1)
            stds = stds.reshape(-1, 1, 1)
            mean_tensor = torch.tensor(means, device=data.device, dtype=torch.float32)
            std_tensor = torch.tensor(stds, device=data.device, dtype=torch.float32)

            # Normalize the data: (data - mean) / std
            data = (data - mean_tensor) / std_tensor

        
        return data.to(dtype=torch.float32), label
