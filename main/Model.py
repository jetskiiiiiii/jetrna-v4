from numpy import transpose
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class BaseAttentionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)

        self.scaling_factor = out_channels**0.5

    def forward(self, in_channels):
        Q = self.W_q(in_channels)
        K = self.W_k(in_channels)
        V = self.W_v(in_channels)

        E = torch.matmul(Q, K.transpose(-2, -1))
        E = E/self.scaling_factor

        A = F.softmax(E, dim=1)

        C = torch.matmul(A, V)

        S_p = torch.cat([Q, C], dim=1)

        return S_p, A

class CNNEncoder(nn.Module):
    """Implementing 1D encoder to extract features from single representation.

    Shape of inputs: (n x 27)

    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Projecting single representation into higher dimension
        self.fc1 = nn.Linear(in_channels, hidden_channels)

        # Feature extraction
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # Decoding layer
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc1(x)

        # Transpose to (batch, 27, n)
        x = x.transpose(1, 2)

        latent = F.leaky_relu(self.conv1(x))
        latent = F.leaky_relu(self.conv2(latent))

        reconstruction = self.conv3(latent)

        # Transpose back to (batch, n, hidden_channels)
        latent = latent.transpose(1, 2)
        reconstruction = reconstruction.transpose(1, 2)

        return latent, reconstruction 

class CoordinatePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_channels)
        self.ffn_linear = nn.Linear(hidden_channels, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.regress_head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # Applying residual block
        #x_res = x
        #x = self.norm(x)
        #x = self.ffn_linear(x)
        #x = F.relu(x)
        #x = x + x_res

        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.regress_head(x)
        return x

class jetRNA_v4_Model(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.bae = BaseAttentionEncoder(in_channels, hidden_channels)
        self.cnn_encoder = CNNEncoder(in_channels, hidden_channels, out_channels)
        self.gnn = CoordinatePredictor(2*hidden_channels, hidden_channels, out_channels)
        self.lr = learning_rate

    def forward(self, data):
        features = self.cnn_encoder(data)
        return features

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, edges, y = batch
        features, reconstructed = self.cnn_encoder(x)         # Features are, for now, the predicted coords
        mask = (x!=-1.0).any(dim=-1)
        reconstructed_real = reconstructed[mask]
        x_real = x[mask]
        loss = F.mse_loss(reconstructed_real, x_real)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, edges, y = batch
        features, reconstructed = self.cnn_encoder(x)         # Features are, for now, the predicted coords
        mask = (x!=-1.0).any(dim=-1)
        reconstructed_real = reconstructed[mask]
        x_real = x[mask]
        loss = F.mse_loss(reconstructed_real, x_real)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, edges, y = batch
        features, _ = self.cnn_encoder(x)
        return features     # Returns (batch, N, out_channels)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

