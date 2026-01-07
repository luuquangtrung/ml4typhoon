import os
import pandas as pd

from argparse import ArgumentParser
from Utils.Map_evaluate import *

parser = ArgumentParser(description="Map evaluate")

parser.add_argument("--temp", dest="temp", required=True)
parser.add_argument("--out", dest="out", required=True)

args = parser.parse_args()

temp = args.temp
out = args.out


# provide the template of the map prediction
csv_temp = temp

cols = ['PRS', 'RCL', 'F1S', 'Predict']
col1 = ['PRS_1', 'RCL_1', 'F1S_1', 'Predict_1']
col0 = ['PRS_0', 'RCL_0', 'F1S_0', 'Predict_0']
t = col1.copy()
t.extend(col0)
scoreboard = pd.DataFrame(columns=t)

for step in range(2, 20, 2):
    print(step)
    csv_path = csv_temp.format(step)
    try:
        pd.read_csv(csv_path)
    except:
        continue

    result = Model_result(csv_path, step_forecast=step)
    df = result.location_acc(dt_list=[
        (datetime(2017, 1, 1), datetime(2023, 1, 1)),
    ])
    
    print(df)

    scoreboard.loc[step, col1] = df.loc['1', cols].values
    scoreboard.loc[step, col0] = df.loc['0', cols].values
    
scoreboard.to_excel(os.path.join(out, "map_eval.xlsx"))
    