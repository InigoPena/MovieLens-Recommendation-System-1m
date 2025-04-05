import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from cf_1m_model import CollaborativeFilteringModel, train_model

def data_preparation(train_df, test_df, val_df):
    """
    Prepare data for the model.
    
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data.
        val_df (pd.DataFrame): Validation data.
    
    Returns:
        DataLoader: DataLoader for training, testing and validation datasets.
    """
    # Convert ratings from 1-5 to 0-4 for PyTorch compatibility
    train_df['Rating'] = train_df['Rating'] - 1
    val_df['Rating'] = val_df['Rating'] - 1
    test_df['Rating'] = test_df['Rating'] - 1

    # Convert DataFrames to PyTorch tensors
    train_tensor = TensorDataset(
        torch.tensor(train_df['User'].values, dtype=torch.long),
        torch.tensor(train_df['Movie'].values, dtype=torch.long),
        torch.tensor(train_df['Rating'].values, dtype=torch.long)
    )
    
    test_tensor = TensorDataset(
        torch.tensor(test_df['User'].values, dtype=torch.long),
        torch.tensor(test_df['Movie'].values, dtype=torch.long),
        torch.tensor(test_df['Rating'].values, dtype=torch.long)
    )
    
    val_tensor = TensorDataset(
        torch.tensor(val_df['User'].values, dtype=torch.long),
        torch.tensor(val_df['Movie'].values, dtype=torch.long),
        torch.tensor(val_df['Rating'].values, dtype=torch.long)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)
    
    return train_loader, test_loader, val_loader

def main():

    train_df = pd.read_csv('ml-1m/train.csv')
    test_df = pd.read_csv('ml-1m/test.csv')
    val_df = pd.read_csv('ml-1m/val.csv')

    train_loader, test_loader, val_loader = data_preparation(train_df, test_df, val_df)
    if train_loader is not None:
        print("Data preparation successful.")
    else:
        print("Data preparation failed.")
        return
    
    cf_model = CollaborativeFilteringModel(n_users=6040, n_movies=3706)

    # Train the model
    train_model(cf_model, train_loader, test_loader)

    # Save the model
    torch.save(cf_model.state_dict(), 'models/cf_model.pth')
    print("Model saved successfully in the 'models' folder as 'cf_model.pth'.")


if __name__ == "__main__":
    main()