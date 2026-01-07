import torch
import os

import numpy as np
import pandas as pd

from typing import List
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

torch.manual_seed(0)

def test_classification(testLoader: DataLoader,
                        model,
                        whole_metrics: list,
                        class_metrics: list,
                        model_path: Path = "Path",
                        out_dir: Path = "Path",
                        device: str = "cuda" if torch.cuda.is_available() else "cpu",
                        export_result = None):
    print(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        pred_list = []
        true_list = []
        score_list = []
        # print(len(testLoader))
        for input, true in tqdm(testLoader):
            if isinstance(input, List):
                input = [hihi.to(device) for hihi in input]
            else:
                input = input.to(device)
            true = true.to(device).type(torch.LongTensor)
            
            pred = model(input)
            pred = pred.squeeze(1)  # remove the extra dimension for binary classification
            
            if len(pred.shape) == 1:
                # for BCEWithLogitsLoss, we need to apply sigmoid before argmax
                pred = torch.sigmoid(pred)
                score_list.extend(pred.cpu().detach().numpy())
                pred = (pred > 0.5)
            else:
                score_list.extend(pred[:, 1].cpu().detach().numpy())
                pred = torch.argmax(pred, dim=1)
            pred_list.extend(pred.cpu().detach().numpy())
            true_list.extend(true.cpu().detach().numpy())
        
        score = np.array(score_list, dtype=float)
        pred = np.array(pred_list, dtype=int)
        true = np.array(true_list, dtype=int)
        # print(np.unique(true))
    
    scoreboard = pd.DataFrame()
    
    if export_result is not None:
        for metric in whole_metrics:
            scoreboard.loc['All', metric.__name__] = metric(true, pred)
        for metric in class_metrics:
            scoreboard.loc['All', metric.__name__] = metric(true, pred)
            for _class in np.unique(true):
                scoreboard.loc[str(_class), metric.__name__] = metric(true, pred, label=_class)
        scoreboard['Support'] = 0
        scoreboard.loc['All', 'Support'] = len(true)
        for _class in np.unique(true):
            scoreboard.loc[str(_class), 'Support'] = len(true[true == _class])
        scoreboard.to_excel(os.path.join(out_dir, f'{export_result}.xlsx'))
        print(scoreboard)
    
    return score