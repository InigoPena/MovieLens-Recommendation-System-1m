# For Classification (e.g., predicting discrete classes like ratings categories):
# Use metrics like Accuracy, Precision, Recall, F1-score, or Confusion Matrix.

import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from cf_1m_model import CollaborativeFilteringModel, prepare_loader
import pandas as pd

def evaluate_model(model, data_loader):
    """
    Evaluate the model on the given DataLoader.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for evaluation data.

    Returns:
        float: Accuracy of the model.
        ndarray: Confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_users, batch_movies, batch_ratings in data_loader:
            # Forward pass
            outputs = model(batch_users, batch_movies)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_ratings.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, conf_matrix

def plot_and_save_results(conf_matrix, accuracy):
    """
    Plot and save the evaluation results.

    Args:
        conf_matrix (ndarray): Confusion matrix.
        accuracy (float): Accuracy of the model.
        results_folder (str): Folder to save the plots.
    """

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='viridis')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.savefig(f'{'results'}/confusion_matrix.png')
    print(f"Confusion matrix saved to {'results'}/confusion_matrix.png")

    # Save accuracy as a text file
    with open(f'{'results'}/accuracy.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}\n')
    print(f"Accuracy saved to {'results'}/accuracy.txt")

if __name__ == "__main__":

    # Load the trained model
    model = CollaborativeFilteringModel(n_users=6040, n_movies=3706)
    model.load_state_dict(torch.load('models/cf_model.pth'))

    val_loader = prepare_loader('ml-1m/val.csv')

    # Evaluate the model
    accuracy, conf_matrix = evaluate_model(model, val_loader)

    # Plot and save results
    plot_and_save_results(conf_matrix, accuracy)