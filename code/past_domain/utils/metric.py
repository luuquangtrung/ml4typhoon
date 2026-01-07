from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(all_labels, all_preds):
    try:
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except ValueError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1