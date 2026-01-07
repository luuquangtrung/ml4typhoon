import os

import pandas as pd
from argparse import ArgumentParser
from math import nan
from pathlib import Path

from sklearn.model_selection import train_test_split

parser = ArgumentParser(description="Prepare data")

parser.add_argument("--path", dest="path", required=True)
parser.add_argument("--step", dest="step", default=2)
parser.add_argument("--ratio", dest="ratio", default=30)
parser.add_argument("--dst", dest="destination", required=True)

args = parser.parse_args()

path = args.path
step = args.step
ratio = args.ratio
dst = args.dst

csv_path = Path(path)
df = pd.read_csv(csv_path)

dataset = {
    'train': [1980, 2016],
    'test': [2017, 2023],
}

# train/val splot by percent

out_dir = Path(dst)
Path(out_dir).mkdir(parents=True, exist_ok=True)
print(out_dir)

# prepare train/val
for ds, irange in dataset.items():
    df['Label'] = nan
    df.loc[
        (df['Year'].between(*irange, inclusive='both')) &
        (df['Position'] == 0) &
        (df['Step']).between(step, step, inclusive='both')
    , 'Label'] = 1
    
    df.loc[
        (df['Year'].between(*irange, inclusive='both')) &
        (df['Position'] == 0) &
        (df['Step']).between(step, 40, inclusive='right')
    , 'Label'] = 0
    
    df.loc[
        (df['Year'].between(*irange, inclusive='both')) &
        (df['Position'] > 0) &
        (df['Step']).between(step, 40, inclusive='both')
    , 'Label'] = 0
    
    df.loc[
        (df['Noise']) &
        (df['Label'] != 1)
    , 'Label'] = nan
    
    if ds == 'train':
        # df.loc[df[df['Label'] == 0].sample((df['Label'] == 0).sum() - (df['Label'] == 1).sum() * ratio).index, 'Label'] = nan
        train, val = train_test_split(df[~ df['Label'].isna()], stratify=df[~ df['Label'].isna()]['Label'], test_size=0.1)
        
        train_df = df.copy()
        train_df.loc[val.index, 'Label'] = nan
        csv_path = os.path.join(out_dir, 'train.csv')
        train_df.to_csv(csv_path, index=False)
        print('train', len(train_df[train_df['Label'] == 1]), len(train_df[train_df['Label'] == 0]), len(train_df[train_df['Label'].isna()]))
        
        val_df = df.copy()
        val_df.loc[train.index, 'Label'] = nan
        csv_path = os.path.join(out_dir, 'val.csv')
        val_df.to_csv(csv_path, index=False)
        print('val', len(val_df[val_df['Label'] == 1]), len(val_df[val_df['Label'] == 0]), len(val_df[val_df['Label'].isna()]))
    
    else:
        csv_path = os.path.join(out_dir, ds + '.csv')
        df.to_csv(csv_path, index=False)
        print('test', len(df[df['Label'] == 1]), len(df[df['Label'] == 0]), len(df[df['Label'].isna()]))
        
        df.loc[df[df['Label'] == 0].sample((df['Label'] == 0).sum() - (df['Label'] == 1).sum() * ratio).index, 'Label'] = nan
        csv_path = os.path.join(out_dir, ds + '2.csv')
        df.to_csv(csv_path, index=False)
        print('test2', len(df[df['Label'] == 1]), len(df[df['Label'] == 0]), len(df[df['Label'].isna()]))