# For Classification (e.g., predicting discrete classes like ratings categories):
# Use metrics like Accuracy, Precision, Recall, F1-score, or Confusion Matrix.

import torch
import os
import seaborn as sns
from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from cf_1m_model import CollaborativeFilteringModel, prepare_loader
import pandas as pd

def evaluate_model(model, data_loader):

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

    # classification report
    classification_rep = classification_report(all_labels, all_preds, output_dict=True)
    classification_rep = pd.DataFrame(classification_rep).T

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, conf_matrix, classification_rep

def plot_and_save_results(conf_matrix, accuracy, classification_rep):

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='viridis')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.savefig(f'{'results'}/confusion_matrix.png')
    print(f"Confusion matrix saved to {'results'}/confusion_matrix.png")

    # Save classification report
    plt.figure(figsize=(10, 6))
    sns.heatmap(classification_rep, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.title('Classification Report')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.tight_layout()
    plt.savefig(f'results/classification_report.png')
    print(f"Classification report saved to results/classification_report.png")

if __name__ == "__main__":

    # Load the trained model
    model = CollaborativeFilteringModel(n_users=6040, n_movies=3706)
    model.load_state_dict(torch.load('models/cf_model.pth'))

    val_loader = prepare_loader('ml-1m/val.csv')

    # Evaluate the model
    accuracy, conf_matrix, classification_rep = evaluate_model(model, val_loader)

    # Plot and save results
    plot_and_save_results(conf_matrix, accuracy, classification_rep)