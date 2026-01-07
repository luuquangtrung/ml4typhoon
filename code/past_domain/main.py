from config import parse_arguments
from data.split_set import split_and_normalize
from data.data_loader import create_dataloader
from models.resnet import Resnet,ResnetContrastive
from utils.class_weight import class_weight
from utils.training import train_model
from utils.evaluating import evaluate_model
from utils.plotting import plot_and_save
from utils.saving_history import save_training_history
from utils.saving_metric import generate_and_save_metrics
import sys
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os



# python main.py --time t2_rus4_cw3_fe --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 4 --class_weight 3 --small_set
# module load python/gpu/3.10.10

def main():
    config = parse_arguments()


    train_dataset, val_dataset, test_dataset = split_and_normalize(
                                                    csv_file=config.csv_path,
                                                    pos_ind=config.pos_ind,
                                                    norm_type=config.norm_type,
                                                    small_set=config.small_set,
                                                    under_sample=config.under_sample,
                                                    rus = config.rus,
                                                )

    train_loader, val_loader, test_loader = create_dataloader(
                                                    train_dataset,
                                                    val_dataset,
                                                    test_dataset,
                                                    shuffle= False
                                                )

    start_time = time.time()
    # Example usage
    for batch_data, batch_labels in train_loader:
        print("Train batch data shape:", batch_data.shape)
        print("Train batch labels shape:", batch_labels.shape)
        break
    end_time = time.time()
    print("Time taken for one batch:", end_time - start_time)

    num_neg, num_pos = (train_dataset.labels == 0).sum(), (train_dataset.labels == 1).sum()
    print(f"Number of 0 labels: {num_neg}")
    print(f"Number of 1 labels: {num_pos}")
    print(f"Number of 0 labels: {(val_dataset.labels == 0).sum()}")
    print(f"Number of 1 labels: {(val_dataset.labels == 1).sum()}")
    print(f"Number of 0 labels: {(test_dataset.labels == 0).sum()}")
    print(f"Number of 1 labels: {(test_dataset.labels == 1).sum()}")
    print(f"Contrastive: {config.contrastive}")

    inp_channel = batch_data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model == 'resnet':
        if config.contrastive == False:
            model = Resnet(inp_channels=inp_channel,
                num_residual_block=[2, 2, 2, 2],
                num_class=1).to(device)
        
        
    if config.under_sample == False:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((num_neg/num_pos)/5)).to(device)
        print(f"Class weights: {torch.tensor((num_neg/num_pos)/5)}")
    
    elif config.class_weight == 1:
        class_weights_tensor = torch.tensor(class_weight(num_pos, num_neg), dtype=torch.float).to(device)
        print(f"Class weights: {class_weights_tensor}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=(class_weights_tensor[1]).float()).to(device)

    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.rus/5)).to(device)
        print(f"Class weights: {torch.tensor(config.rus/5)}")
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
    }

    model = train_model(model, 
                        device, 
                        train_loader, 
                        val_loader,
                        test_loader, 
                        criterion, 
                        optimizer, 
                        config.epoch, 
                        config.time,
                        history,
                        config.contrastive)

    save_training_history(history, config.time, config.epoch)
    
    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
        plot_and_save(history, metric, config.time, metric.capitalize())

    # Evaluate the model on the test set
    all_preds, all_labels, test_loss, test_accuracy = evaluate_model(model, 
                                                                     test_loader, 
                                                                     test_dataset, 
                                                                     criterion, 
                                                                     device, 
                                                                     config.time,
                                                                     config.contrastive
                                                                     )

    # Generate confusion matrix and metrics
    y_pred_binary = np.array([pred.cpu().numpy() for pred in all_preds]).astype(int)
    y_true = np.array(all_labels).astype(int)
    generate_and_save_metrics(y_true, y_pred_binary, config.time)

    model_dir = f'./result/model'
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), f'{model_dir}/trained_model_{config.time}.pth')

if __name__ == "__main__":
    main()
