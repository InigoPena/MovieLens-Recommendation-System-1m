import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from cf_1m_model import CollaborativeFilteringModel, train_model, prepare_loader
from cf_1m_regression_model import CollaborativeFilteringRegression, train_regression_model, prepare_loader_for_regression

def main():

    train_loader = prepare_loader('ml-1m/train.csv')
    val_loader = prepare_loader('ml-1m/val.csv')
    
    if train_loader is not None:
        print("Data preparation successful.")
    else:
        print("Data preparation failed.")
        return
    
    cf_model = CollaborativeFilteringModel(n_users=6040, n_movies=3706)

    # Train the model
    train_model(cf_model, train_loader, val_loader)

    # Save the model
    torch.save(cf_model.state_dict(), 'models/cf_model.pth')
    print("Model saved successfully in the 'models' folder as 'cf_model.pth'.")
    
    ###==================================================================================================================
    # Regression Model Training
    ###==================================================================================================================


    cf_regression_model = CollaborativeFilteringRegression(n_users=6040, n_movies=3706)

    train_loader_regression = prepare_loader_for_regression('ml-1m/train.csv')
    val_loader_regression = prepare_loader_for_regression('ml-1m/val.csv')

    # Train the regression model
    train_regression_model(cf_regression_model, train_loader_regression, val_loader_regression)

    # Save the regression model
    torch.save(cf_regression_model.state_dict(), 'models/cf_regression_model.pth')
    print("Regression model saved successfully in the 'models' folder as 'cf_regression_model.pth'.")



if __name__ == "__main__":
    main()