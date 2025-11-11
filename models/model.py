import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ResultsHandler:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.results = []

    def update_results(self, epoch, loss, accuracy, labels, preds):
        self.results.append({"epoch": epoch, "loss": loss, "accuracy": accuracy, "labels": labels, "preds": preds})

    def plot_confusion_matrix(self, true_labels, pred_labels, detection, classification):
        """Plots and saves a confusion matrix."""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
        
        task = "Detection" if detection else "Classification"
        plt.title(f"{task} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        # Save confusion matrix
        os.makedirs("confusion_matrices", exist_ok=True)
        filename = f"confusion_matrices/{task.lower()}_confusion_matrix.png"
        plt.savefig(filename)
        print(f"Confusion matrix saved as {filename}")

        plt.show()
    
