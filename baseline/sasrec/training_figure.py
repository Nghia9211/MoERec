import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(loss_history, l2_history, save_path, val_loss_history=None):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)

    epochs = range(1, len(loss_history) + 1)
    
    plt.plot(epochs, loss_history, label='Train Loss', color='blue', marker='o', markersize=3)

    if val_loss_history is not None:
        plt.plot(epochs, val_loss_history, label='Val Loss', color='red', linestyle='--', marker='s', markersize=3)
        plt.title('Training & Validation Loss')
    else:
        plt.title('Training Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, l2_history, label='L2 Regularization', color='orange', marker='^', markersize=3)
    plt.title('Embedding L2 Norm (Regularization Check)')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Value')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300) 
    plt.close()