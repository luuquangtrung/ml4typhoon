import sys
import os

import numpy as np

from argparse import ArgumentParser
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight

from Dataset.Merra2_dataset import *
from Model.PointOut.ResNet_classification import *
from Progress.Train_progress import *
from Progress.Test_progress import *
from Utils.Metrics import *

print('Importing Merra2 dataset and ResNet model for classification...')

parser = ArgumentParser(description="Input Data")
parser.add_argument("-p", "--path",dest="config_path", required=True)
args = parser.parse_args()

config = json.load(open(args.config_path, 'r'))

inp_dir = config['inp_dir']
out_dir = config['out_dir']
agg_step = config['agg_step']
pos_weight = config['pos_weight']

version = 0
while True:
    if os.path.isdir(out_dir + f'_v{version}'):
        version += 1
    else:
        # version = 1
        out_dir = out_dir + f'_v{version}'
        break
    
Path(out_dir).mkdir(parents=True, exist_ok=True)

#===================
# Prepare data phase
#===================
print(f'Input directory: {inp_dir}')
trainLoader, valLoader, testLoader = prepare_Dataset(
    DatasetClass = Merra2_dataset,
    train_path = os.path.join(inp_dir, 'train.csv'),
    val_path = os.path.join(inp_dir, 'val.csv'),
    test_path = os.path.join(inp_dir, 'test.csv'),
    agg_step = agg_step,
    batch_size = 32,
    num_workers = 16,
    out_dir = out_dir,
    pre_load = True,
)

#====================
# Prepare model phase
#====================
model = create_model(out_dir = out_dir,
                     inp_channels = INP_CHANNELS,
                     num_residual_block = [2, 2, 2, 2], 
                     num_class = 2,)

#============
# Train phase
#============

if pos_weight > 0:
    class_weight = np.array([
        (pos_weight + 1) / (2 * pos_weight),
        (pos_weight + 1) / 2
    ])
else:
    true_list = []
    for _, true in trainLoader:
        true_list.extend(true.cpu().detach().numpy())
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(true_list), y=true_list)

# class_weight = np.array([0.75, 1.5])

    
train_classification(trainLoader = trainLoader,
                     valLoader = valLoader,
                     model = model,
                     loss_func = CrossEntropyLoss,
                     class_weight = class_weight,
                     optimizer = Adam,
                     learning_rate = 1e-4,
                     epochs = 100,
                     out_dir = out_dir)

test_classification(testLoader = testLoader,
                    model = model,
                    whole_metrics = [ACC],
                    class_metrics = [PRS, RCL, F1S],
                    model_path = os.path.join(out_dir, 'model.pth'),
                    export_result = 'test_score',
                    out_dir = out_dir)

trainLoader, valLoader, testLoader = prepare_Dataset(
    DatasetClass = Merra2_dataset,
    train_path = 'any_path.csv',
    val_path = 'any_path.csv',
    test_path = os.path.join(inp_dir, 'test2.csv'),
    agg_step = agg_step,
    batch_size = 32,
    num_workers = 16,
    pre_load = False,
)

test_classification(testLoader = testLoader,
                    model = model,
                    whole_metrics = [ACC],
                    class_metrics = [PRS, RCL, F1S],
                    model_path = os.path.join(out_dir, 'model.pth'),
                    export_result = 'test2_score',
                    out_dir = out_dir)

trainLoader, valLoader, testLoader = prepare_Dataset(
    DatasetClass = Merra2_dataset,
    train_path = 'any_path.csv',
    val_path = 'any_path.csv',
    test_path = 'map_test.csv',
    agg_step = agg_step,
    batch_size = 32,
    num_workers = 16,
    # out_dir = out_dir,
    pre_load = False,
)

score = test_classification(testLoader = testLoader,
                    model = model,
                    whole_metrics = [ACC],
                    class_metrics = [PRS, RCL, F1S],
                    model_path = os.path.join(out_dir, 'model.pth'),
                    # export_result = 'map_test',
                    out_dir = out_dir)

# print(score.shape)
df = pd.read_csv('map_test.csv')
# print(df.shape)
df.loc[~ df['Label'].isna(), 'Score'] = score
df.to_csv(os.path.join(out_dir, 'map_test.csv'), index=False)