import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def ACC(true, pred):
    return accuracy_score(true, pred)

def PRS(true, pred, label=-1):
    if label < 0:
        return precision_score(true, pred, average='macro')
    else:
        return precision_score(true == label, pred == label)

def RCL(true, pred, label=-1):
    if label < 0:
        return recall_score(true, pred, average='macro')
    else:
        return recall_score(true == label, pred == label)

def F1S(true, pred, label=-1):
    if label < 0:
        return f1_score(true, pred, average='macro')
    else:
        return f1_score(true == label, pred == label)

def CC(true, pred):
    true = np.array(true).reshape(-1)
    pred = np.array(pred).reshape(-1)
    return np.corrcoef(true, pred)[0, 1]

def MAE(true, pred):
    true = np.array(true).reshape(-1)
    pred = np.array(pred).reshape(-1)
    return mean_absolute_error(true, pred)

def RMSE(true, pred):
    true = np.array(true).reshape(-1)
    pred = np.array(pred).reshape(-1)
    return root_mean_squared_error(true, pred)

def R2(true, pred):
    true = np.array(true).reshape(-1)
    pred = np.array(pred).reshape(-1)
    return r2_score(true, pred)