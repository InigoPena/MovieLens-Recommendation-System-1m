import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import matplotlib.pyplot as plt

class CollaborativeFilteringRegression(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50, hidden_layers=[32,16]):
        super().__init__()

        # User and movie embeddings
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users,
            embedding_dim=n_factors
            )
        
        self.movie_embedding = nn.Embedding(
            num_embeddings=n_movies,
            embedding_dim=n_factors
            )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        # Dropout layer in case of overfitting
        self.dropout = nn.Dropout(p=0.1)

        # Hidden layers
        layers = []
        input_size = n_factors * 2
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.05)
            ])
            input_size = hidden_size

        # Final output layer with a single neuron for regression
        layers.append(nn.Linear(input_size, 1))
        
        # Optional sigmoid activation to constrain output between 0 and 1
        # It is scaled back to the 1-5 range later
        layers.append(nn.Sigmoid())

        # Create the sequential model
        self.network = nn.Sequential(*layers)

        print(f"Regression model initialized with {n_users} users and {n_movies} movies.\n")

    # Define how data flows through the model
    def forward(self, users, movies):
        user_embed = self.user_embedding(users)
        movie_embed = self.movie_embedding(movies)
        
        # Concatenate embeddings
        x = torch.cat([user_embed, movie_embed], dim=1)
        
        x = self.dropout(x)
        return self.network(x)

# Model Training function for regression
def train_regression_model(model, train_loader, val_loader, epochs=7, learning_rate=0.01):
    
    # Use MSE Loss for regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 3
    counter = 0
    best_model_state = None

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch_users, batch_movies, batch_ratings in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_users, batch_movies)
            outputs = outputs.squeeze() # Remove extra dimension
            
            # Scale ratings from 0-1 to 0-4 range for loss calculation
            scaled_outputs = outputs * 4
            batch_ratings = batch_ratings.float()
            
            # loss
            loss = criterion(scaled_outputs, batch_ratings)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
    
        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_users, batch_movies, batch_ratings in val_loader:
                outputs = model(batch_users, batch_movies).squeeze()
                scaled_outputs = outputs * 4
                batch_ratings = batch_ratings.float()
                val_loss = criterion(scaled_outputs, batch_ratings)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print training and validation loss
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save the best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model loaded.")

    # Plot and save the loss histogram
    os.makedirs('results', exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('results/regression_loss_plot.png')
    print("Loss plot saved to 'results/regression_loss_plot.png'")
    
    return train_losses, val_losses

def prepare_loader_for_regression(file_path, batch_size=64, shuffle=True):
    # Load the data
    df_for_loader = pd.read_csv(file_path)

    # Convert ratings from 1-5 to 0-4 for scaling compatibility
    df_for_loader['Rating'] = df_for_loader['Rating'] - 1

    # Convert DataFrame to PyTorch tensor
    tensor = TensorDataset(
        torch.tensor(df_for_loader['User'].values, dtype=torch.long),
        torch.tensor(df_for_loader['Movie'].values, dtype=torch.long),
        torch.tensor(df_for_loader['Rating'].values, dtype=torch.long)
    )

    # Create DataLoader
    loader = DataLoader(tensor, batch_size=batch_size, shuffle=shuffle)

    return loader