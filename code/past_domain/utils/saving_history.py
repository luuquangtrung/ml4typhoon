import os

def save_training_history(history, time_, num_epochs):
    os.makedirs('./result/history', exist_ok=True)
    with open(f'./result/history/training_history_{time_}.txt', 'w') as f:
        f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")
        for epoch in range(num_epochs):
            if epoch > len(history['loss']):
                continue
            f.write(f"{epoch+1},{history['loss'][epoch]},{history['accuracy'][epoch]},"
                    f"{history['precision'][epoch]},{history['recall'][epoch]},{history['f1'][epoch]},"
                    f"{history['val_loss'][epoch]},{history['val_accuracy'][epoch]},"
                    f"{history['val_precision'][epoch]},{history['val_recall'][epoch]},"
                    f"{history['val_f1'][epoch]}\n")
