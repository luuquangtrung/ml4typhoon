import os
import torch
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader

from Utils.Metrics import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_classification(trainLoader: DataLoader,
                         valLoader: DataLoader,
                         model,
                         class_weight = None,
                         loss_func = CrossEntropyLoss,
                         optimizer = Adam,
                         learning_rate: float = 1e-5,
                         epochs: int = 100,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu",
                         out_dir: Path = 'Path'):
    
    model = model.to(device)
    if class_weight is None:
        loss_func = loss_func().to(device)
    else:
        if loss_func == CrossEntropyLoss:
            loss_func = loss_func(weight=torch.tensor(class_weight, dtype=torch.float)).to(device)
        elif loss_func == BCEWithLogitsLoss:
            # for BCEWithLogitsLoss, class_weight should be a tensor of the same size as the output
            loss_func = loss_func(pos_weight=torch.tensor(class_weight, dtype=torch.float)).to(device)
        
    optimizer = optimizer(model.parameters(),
                          lr = learning_rate)
    # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    
    log = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_prs': [],
        'val_prs': [],
        'train_rcl': [],
        'val_rcl': [],
        'train_f1s': [],
        'val_f1s': [],
    }
    
    for dir in ['model_save', 'plot_train']:
        Path(os.path.join(out_dir, dir)).mkdir(parents=True, exist_ok=True)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        
        totalTrainLoss = 0
        totalValLoss = 0
        
        pred_list = []
        true_list = []
        
        for i, (input, true) in tqdm(enumerate(trainLoader)):
            if isinstance(input, List):
                input = [hihi.to(device) for hihi in input]
            else:
                input = input.to(device)
            true = true.type(torch.LongTensor).to(device)
            # print(true.min(),true.max())
            
            pred = model(input)
            pred = pred.squeeze(1)  # remove the extra dimension for binary classification
            # print(pred.is_cuda)
            # print(true.is_cuda)
            loss = loss_func(pred, true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        
            totalTrainLoss += loss
            if len(pred.shape) == 1:
                # for BCEWithLogitsLoss, we need to apply sigmoid before argmax
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5)
                
            else:
                pred = torch.argmax(pred, dim=1)
            pred_list.extend(pred.cpu().detach().numpy())
            true_list.extend(true.cpu().detach().numpy())
        
        avgTrainLoss = totalTrainLoss / len(trainLoader)
        log['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
        pred_list = np.array(pred_list, dtype=int)
        true_list = np.array(true_list, dtype=int)
        log['train_acc'].append(ACC(true_list, pred_list))
        log['train_prs'].append(PRS(true_list, pred_list))
        log['train_rcl'].append(RCL(true_list, pred_list))
        log['train_f1s'].append(F1S(true_list, pred_list))
        
        with torch.no_grad():
            model.eval()
            
            pred_list = []
            true_list = []
        
            for (i, (input, true)) in tqdm(enumerate(valLoader)):
                # Move data to configured DEVICE
                if isinstance(input, List):
                    input = [hihi.to(device) for hihi in input]
                else:
                    input = input.to(device)
                true = true.type(torch.LongTensor).to(device)
                
                pred = model(input)
                pred = pred.squeeze(1)  # remove the extra dimension for binary classification
                loss = loss_func(pred, true)
                
                totalValLoss += loss
                if len(pred.shape) == 1:
                    # for BCEWithLogitsLoss, we need to apply sigmoid before argmax
                    pred = torch.sigmoid(pred)
                    pred = (pred > 0.5)
                    
                else:
                    pred = torch.argmax(pred, dim=1)
                pred_list.extend(pred.cpu().detach().numpy())
                true_list.extend(true.cpu().detach().numpy())
        
        avgValLoss = totalValLoss / len(valLoader)
        log['val_loss'].append(avgValLoss.cpu().detach().numpy())
        pred_list = np.array(pred_list, dtype=int)
        true_list = np.array(true_list, dtype=int)
        log['val_acc'].append(ACC(true_list, pred_list))
        log['val_prs'].append(PRS(true_list, pred_list))
        log['val_rcl'].append(RCL(true_list, pred_list))
        log['val_f1s'].append(F1S(true_list, pred_list))
        
        # print the model training and validation information
        print("[INFO]: EPOCH {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.4f}, Test loss: {:.4f}".format(avgTrainLoss, avgValLoss))
        
    df = pd.DataFrame(log)
    df.to_excel(os.path.join(out_dir,
                                'train_log.xlsx'))
    
    torch.save(model.state_dict(), os.path.join(out_dir,
                                                f'model.pth'))
    
    for metric in ['loss', 'acc', 'prs', 'rcl', 'f1s']:
        # plot the training loss
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(log["train_" + metric], label="train_" + metric)
        plt.plot(log["val_" + metric], label="val_" + metric)
        plt.title("Training " + metric + " on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel(metric)
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(out_dir, f'plot_train/{metric}.png'))
        plt.close()