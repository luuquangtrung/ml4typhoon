import os

import pandas as pd 

from pathlib import Path
from tqdm import tqdm

rus = 30
weight = 6
agg = 0
suffix = ''
out_temp = 
version = 2
step_list = range(2, 20, 2)

cols = ['PRS', 'RCL', 'F1S']
col1 = ['PRS_1', 'RCL_1', 'F1S_1']
col0 = ['PRS_0', 'RCL_0', 'F1S_0']
t = col1.copy()
t.extend(col0)
scoreboard = pd.DataFrame(columns=t)

fileget = 'test_score'

for step in step_list:
    print(step)
    xlsx_path = os.path.join(
        Path(f'{out_temp}_{str(step)}_v{version}'),
        f'{fileget}.xlsx',
    )
    try:
        df = pd.read_excel(xlsx_path, index_col=0)
    except:
        continue
    scoreboard.loc[step, col1] = df.loc['1', cols].values
    scoreboard.loc[step, col0] = df.loc['0', cols].values
    
scoreboard.to_excel()
