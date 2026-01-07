import torch
import time
from utils.metric import compute_metrics
import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from pytorch_metric_learning.losses import NTXentLoss
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device, contrastive = False):
    if contrastive:
        contrastive_loss = NTXentLoss(temperature=0.10).to(device)
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        
        if contrastive:
            outputs, embs = model(inputs)
            # print(labels.shape)
            loss1 = criterion(outputs, labels)
            loss2 = contrastive_loss(embs,labels.squeeze())
            loss = loss1+loss2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum()
        total += labels.size(0)

    precision, recall, f1 = compute_metrics(all_labels, all_preds)
    return total_loss / len(train_loader), correct / total, precision, recall, f1

def validate_one_epoch(model, validation_loader, criterion, device, contrastive = False):
    if contrastive:
        contrastive_loss = NTXentLoss(temperature=0.10).to(device)
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            if contrastive:
                outputs, embs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = contrastive_loss(embs,labels.squeeze())
                loss = loss1+loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum()
            total += labels.size(0)

    precision, recall, f1 = compute_metrics(all_labels, all_preds)
    return total_loss / len(validation_loader), correct / total, precision, recall, f1

def train_model(model, device, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, time_, history, contrastive):
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    min_delta = 0
    flag = True
    best_model = copy.deepcopy(model)

    early_stopping_path = './result_earlystopping'
    os.makedirs(f'{early_stopping_path}/model', exist_ok=True)
    os.makedirs(f'{early_stopping_path}/metrics', exist_ok=True)
    best_model_path = f'{early_stopping_path}/model/trained_model_{time_}_best.pth'
    
    print(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,contrastive)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device, contrastive)
        
        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['precision'].append(train_prec)
        history['recall'].append(train_rec)
        history['f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.2f}, ' 
              f'Train accuracy: {train_acc:.2f}, '
              f'Train precision: {train_prec:.2f}, '
              f'Train recall: {train_rec:.2f}, '
              f'Train F1: {train_f1:.2f}, '
              f'Validation Loss: {val_loss:.2f}, '
              f'Val accuracy: {val_acc:.2f}, '
              f'Val precision: {val_prec:.2f}, '
              f'Val recall: {val_rec:.2f}, '
              f'Val F1: {val_f1:.2f}'
        )
        
        
        # Implement early stopping or save best model here
        current_val_loss = val_loss
    
        if best_val_loss - current_val_loss > min_delta:
        # if False:
            best_val_loss = current_val_loss
            trigger_times = 0  # Reset the trigger times when improvement occurs
        else:
            if flag == False:
                continue
            trigger_times += 1
            print(f"Early stopping trigger: {trigger_times}/{patience}")
            if trigger_times >= patience:
            # if True:
                print("Early stopping!")
                # break  # Stop training if patience is exceeded
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved new best model with validation loss: {best_val_loss:.4f}')
                flag = False
        
        end_time = time.time()
        print(f"Time one epoch: {end_time - start_time:.2f} seconds")

    ### Eval best model
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")

    else:
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = best_model(inputs)
                # classification_loss = classification_criterion(outputs, labels)
                # test_loss += classification_loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                correct += (preds.squeeze() == labels.cpu().numpy().astype(int)).sum().item()


        # Generate confusion matrix and classification metrics
        y_pred_binary = np.array(all_preds).astype(int)
        y_true = np.array(all_labels).astype(int)

        cm = confusion_matrix(y_true, y_pred_binary)

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        os.makedirs(f'{early_stopping_path}/metrics/confusion_matrix/', exist_ok=True)
        plt.savefig(f'{early_stopping_path}/metrics/confusion_matrix/confusion_matrix_{time_}.png')
        plt.close()


        # Calculate precision, recall, and F1 score
        precision_label_1 = precision_score(y_true, y_pred_binary, pos_label=1)
        recall_label_1 = recall_score(y_true, y_pred_binary, pos_label=1)
        f1_score_label_1 = f1_score(y_true, y_pred_binary, pos_label=1)

        precision_label_0 = precision_score(y_true, y_pred_binary, pos_label=0)
        recall_label_0 = recall_score(y_true, y_pred_binary, pos_label=0)
        f1_score_label_0 = f1_score(y_true, y_pred_binary, pos_label=0)

        precision_overall = precision_score(y_true, y_pred_binary, average='weighted')
        recall_overall = recall_score(y_true, y_pred_binary, average='weighted')
        f1_overall = f1_score(y_true, y_pred_binary, average='weighted')

        # Save metrics to a file
        os.makedirs(f'{early_stopping_path}/metrics/', exist_ok=True)

        file_path = f'{early_stopping_path}/metrics/metrics_{time_}.txt'

        with open(file_path, 'w') as f:
            f.write("Metrics for Label 1:\n")
            f.write(f"Precision: {precision_label_1:.4f}\n")
            f.write(f"Recall: {recall_label_1:.4f}\n")
            f.write(f"F1 Score: {f1_score_label_1:.4f}\n")
            
            f.write("\nMetrics for Label 0:\n")
            f.write(f"Precision: {precision_label_0:.4f}\n")
            f.write(f"Recall: {recall_label_0:.4f}\n")
            f.write(f"F1 Score: {f1_score_label_0:.4f}\n")
            
            f.write("\nOverall Metrics:\n")
            f.write(f"Precision: {precision_overall:.4f}\n")
            f.write(f"Recall: {recall_overall:.4f}\n")
            f.write(f"F1 Score: {f1_overall:.4f}\n")

        print(f'Metrics saved to {file_path}')



    return model