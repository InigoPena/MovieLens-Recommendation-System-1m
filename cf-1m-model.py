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
        
        self.user_embedding.weight.data *= 1e-6
        self.movie_embedding.weight.data *= 1e-6

        # Dropout layer

        # Hidden layers

        # Final output layer