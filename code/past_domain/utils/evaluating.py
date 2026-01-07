import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from pytorch_metric_learning.losses import NTXentLoss

def evaluate_model(model, test_loader, test_dataset, criterion, device, time_,contrastive = False):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    if contrastive:
        contrastive_loss = NTXentLoss(temperature=0.10).to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            if contrastive:
                outputs, embs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = contrastive_loss(embs,labels.squeeze())
                loss = loss1+loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            test_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum()

    test_accuracy = correct / len(test_dataset)
    print("Test Loss:", test_loss / len(test_loader))
    print("Test Accuracy:", test_accuracy)

    return all_preds, all_labels, test_loss, test_accuracy
