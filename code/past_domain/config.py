import argparse
from dataclasses import dataclass

@dataclass
class Config:
    time: str
    pos_ind: int
    norm_type: str
    lr: float
    csv_path: str
    small_set: bool
    model: str
    under_sample: bool
    rus: int 
    class_weight: int 
    alpha: float = 0.85
    epoch: int = 100
    contrastive: bool = False
    feature_expert: bool = False
    

def parse_arguments() -> Config:
    parser = argparse.ArgumentParser(description="Train and Evaluate Model")
    parser.add_argument('--time', type=str, required=True, help='Time value (e.g., 1_FE)')
    parser.add_argument('--pos_ind', type=int, required=False, default=1, help='Positive step')
    parser.add_argument('--norm_type', type=str, required=True, help='Type of normalization (new/old)')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    parser.add_argument('--small_set', action='store_true', help='Use small set for testing')   
    parser.add_argument('--model', type=str, default='resnet', help='Model type')
    parser.add_argument('--under_sample', action='store_true', help='Under sample data')
    parser.add_argument('--rus', type=int, required=False, help='Undersample ratio')
    parser.add_argument('--class_weight', type=int, required=False, help='Class weight')
    args = parser.parse_args()

    # Ex: python main.py --time t2_rus4_cw3_fe_back --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 4 --class_weight 3 --small_set

    args.csv_path = f"csv/merra_full_new.csv"

    config = Config(
        time=args.time,
        pos_ind=args.pos_ind,
        norm_type=args.norm_type,
        lr=args.lr,
        csv_path=args.csv_path,
        small_set=args.small_set,
        model=args.model,
        under_sample=args.under_sample,
        rus=args.rus,
        class_weight=args.class_weight,
    )

    if config.small_set:
        config.epoch = 5

    print(f'time: {config.time}')
    print(f'pos_ind: {config.pos_ind}')
    print(f'norm_type: {config.norm_type}')
    print(f'lr: {config.lr}')
    print(f'csv_path: {config.csv_path}')
    print(f'alpha: {config.alpha}')
    print(f'epoch: {config.epoch}')
    print(f'small_set: {config.small_set}')
    print(f'model: {config.model}')
    print(f'under_sample: {config.under_sample}')
    print(f'rus: {config.rus}')
    print(f'class_weight: {config.class_weight}')

    return config