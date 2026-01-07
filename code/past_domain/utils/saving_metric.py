from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_and_save_metrics(y_true, y_pred_binary, time_):
    cm = confusion_matrix(y_true, y_pred_binary)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    os.makedirs('./result/confusion_matrix/', exist_ok=True)
    plt.savefig(f'./result/confusion_matrix/confusion_matrix_{time_}.png')
    plt.close()

    precision_label_1 = precision_score(y_true, y_pred_binary, pos_label=1)
    recall_label_1 = recall_score(y_true, y_pred_binary, pos_label=1)
    f1_score_label_1 = f1_score(y_true, y_pred_binary, pos_label=1)

    precision_label_0 = precision_score(y_true, y_pred_binary, pos_label=0)
    recall_label_0 = recall_score(y_true, y_pred_binary, pos_label=0)
    f1_score_label_0 = f1_score(y_true, y_pred_binary, pos_label=0)

    precision_overall = precision_score(y_true, y_pred_binary, average='weighted')
    recall_overall = recall_score(y_true, y_pred_binary, average='weighted')
    f1_overall = f1_score(y_true, y_pred_binary, average='weighted')

    os.makedirs('./result/metrics/', exist_ok=True)
    file_path = f'./result/metrics/metrics_{time_}.txt'

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
