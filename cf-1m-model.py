import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CollaborativeFilteringModel(nn.Module):
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

        # # Scale the weights
        # self.user_embedding.weight.data *= 1e-6
        # self.movie_embedding.weight.data *= 1e-6

        # Dropout layer in case of overfitting
        self.dropout = nn.Dropout(p=0.05)

        # Hidden layers
        layers = []
        input_size = n_factors * 2
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.05)
            ])
            input_size = hidden_size

        # Final output layer with softmax (for multiclass clasification) activation
        layers.append(nn.Linear(input_size, 5))
        layers.append(nn.Softmax(dim=1))

        # Create the sequential model
        self.network = nn.Sequential(*layers)

    # Define how data flows through the model
    def forward(self, users, movies):
        
        # Embed users and movies
        user_embed = self.user_embedding(users)
        movie_embed = self.movie_embedding(movies)
        
        # Concatenate embeddings
        x = torch.cat([user_embed, movie_embed], dim=1)
        
        # Pass through dropout and network
        x = self.dropout(x)
        return self.network(x)

# Function to train the model
def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler (similar to Keras ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.75, 
        patience=3, 
        min_lr=1e-6, 
        verbose=True
    )

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch_users, batch_movies, batch_labels in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_users, batch_movies)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
    
        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            model.eval()
            for batch_users, batch_movies, batch_labels in test_loader:
                outputs = model(batch_users, batch_movies)
                val_loss = criterion(outputs, batch_labels)
                total_val_loss += val_loss.item()
