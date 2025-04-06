# For Classification (e.g., predicting discrete classes like ratings categories):
# Use metrics like Accuracy, Precision, Recall, F1-score, or Confusion Matrix.

import torch
import os
import seaborn as sns
from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from cf_1m_model import CollaborativeFilteringModel, prepare_loader
from cf_1m_regression_model import CollaborativeFilteringRegression, prepare_loader_for_regression
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

def evaluate_regression_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_users, batch_movies, batch_ratings in test_loader:
            outputs = model(batch_users, batch_movies).squeeze()
            
            # Scale outputs from 0-1 to 0-4 range
            scaled_outputs = outputs * 4
            
            # Convert back to 1-5 range for reporting
            final_predictions = scaled_outputs + 1
            actual_ratings = batch_ratings + 1
            
            all_predictions.extend(final_predictions.tolist())
            all_targets.extend(actual_ratings.tolist())
    
    predictions = torch.tensor(all_predictions)
    targets = torch.tensor(all_targets)
    
    # Calculate mse
    mse = ((predictions - targets) ** 2).mean().item()
    
    # Calculate rmse
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # Calculate mae
    mae = torch.abs(predictions - targets).mean().item()
    
    return mse, rmse, mae

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

    ###===================================================================================================================
    # Regression Model Evaluation
    ###===================================================================================================================

    model_regression = CollaborativeFilteringRegression(n_users=6040, n_movies=3706)
    model_regression.load_state_dict(torch.load('models/cf_regression_model.pth'))

    val_loader_regression = prepare_loader_for_regression('ml-1m/val.csv')

    # Evaluate the regression model
    mse, rmse, mae = evaluate_regression_model(model_regression, val_loader_regression)
    print(f"Regression Model Evaluation:\n MSE: {mse:.4f}\n RMSE: {rmse:.4f}\n MAE: {mae:.4f}")