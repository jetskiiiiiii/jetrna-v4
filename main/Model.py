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
        self.conv3 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

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


class GNNCoordinatePredictor(nn.Module):
    """GCN based decoder to predict final coordinates of sequence.

    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class jetRNA_v4_Model(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.cnn_encoder = CNNEncoder(in_channels, hidden_channels, hidden_channels)
        self.gnn_predictor = GNNCoordinatePredictor(hidden_channels, hidden_channels, out_channels)
        self.lr = learning_rate

    def forward(self, x, edges):
        batch_size, seq_len, _ = x.shape
        
        # Extract features from CNN
        features, reconstruction = self.cnn_encoder(x)

        # Flatten for GCN
        x_flat = features.reshape(-1, features.size(-1))

        # Creating combined edge_index for batch
        combined_edges = []
        for i, edge in enumerate(edges):
            combined_edges.append(edge + (i * seq_len))
        batched_edge_index = torch.cat(combined_edges, dim=1)

        # Predicting coordinates with GCN
        pred_coords_flat = self.gnn_predictor(x_flat, batched_edge_index)

        pred_coords = pred_coords_flat.view(batch_size, seq_len, 3)

        return pred_coords, reconstruction

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, "val")
        return loss

    def shared_step(self, batch, step):
        x, edges, y, y_center = batch
        pred_coords, reconstructed = self(x, edges)         # Features are, for now, the predicted coords

        mask = (x!=-1.0).any(dim=-1)                        # Ignoring padding

        loss_rec = F.mse_loss(reconstructed[mask], x[mask])
        loss_coord = F.mse_loss(pred_coords[mask], y[mask])

        # Calculating bond length loss
        bond_loss = 0
        ideal_dist = 5.0 / 900.0    # 5 Angstroms
        for i in range(pred_coords.shape[0]):
            seq_coords = pred_coords[i]
            seq_mask = mask[i]

            real_coords = seq_coords[seq_mask]

            if len(real_coords) > 1:
                diffs = real_coords[1:] - real_coords[:-1]
                dists = torch.norm(diffs, dim=-1)

                bond_loss += F.mse_loss(dists, torch.full_like(dists, ideal_dist))

        total_loss = loss_coord + (0.5 * bond_loss) + (0.1 * loss_rec)

        self.log(f"{step}_loss_coord", loss_coord)
        return total_loss

    def predict_step(self, batch, batch_idx):
        x, edges, y, y_center = batch
        pred_coords, reconstructed = self(x, edges)         # Features are, for now, the predicted coords
        return pred_coords, reconstructed, y_center         # Returns coords - (); reconstructed - (batch, N, hidden_channels)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

