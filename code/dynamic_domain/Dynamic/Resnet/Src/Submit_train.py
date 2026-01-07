import os
import json

from pathlib import Path

rus = 30
agg_step = 0
pos_weight = 6
# corr = 0.9
out_temp = f'path-to-output-dir_template'
dataset_temp = f'path-to-input-dataset_template'
train_script = f'training-script'


for step in range(2, 4, 2):
    print(f'Processing step {step}...')
    out_dir = Path(f'{out_temp}_{str(step)}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'inp_dir': f'{dataset_temp}_{str(step)}',
        'out_dir': f'{out_temp}_{str(step)}',
        'agg_step': agg_step,
        'pos_weight': pos_weight,
        # 'corr':corr
    }
    
    json.dump(
        config,
        open(os.path.join(out_dir, 'config.json'), 'w'),
    )
    
    print(f'Config file saved to {os.path.join(out_dir, "config.json")}')
    os.system(f'python {train_script} -p {os.path.join(out_dir, "config.json")}')
    continue
    
    with open(os.path.join(out_dir, 'submit.sh'), 'w') as bash:
        bash.write(f'''\
#!/bin/bash -l
#SBATCH -D 
#SBATCH -A r00043
#SBATCH -J HiR{rus}W{pos_weight}_{step:02d}
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu 
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=240G
#SBATCH --cpus-per-gpu=64
cd {out_dir}
echo {out_dir}
module load python/gpu/3.10.10
python {train_script} -p {os.path.join(out_dir, 'config.json')}
        ''')
    
    sh_path = os.path.join(out_dir, 'submit.sh')
    os.system(f'sbatch {sh_path}')