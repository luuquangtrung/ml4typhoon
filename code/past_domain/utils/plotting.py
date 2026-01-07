import matplotlib.pyplot as plt
import os 

def plot_and_save(history, metric, time_, metric_name):
    plt.figure()
    plt.plot(history[metric], label=f'Train {metric_name}')
    plt.plot(history[f'val_{metric}'], label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    os.makedirs(f'./result/history/metrics/{metric_name.lower()}', exist_ok=True)
    plt.savefig(f'./result/history/metrics/{metric_name.lower()}/{metric_name.lower()}_plot_{time_}.png')
    plt.close()
