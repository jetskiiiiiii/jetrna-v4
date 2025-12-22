import torch
import lightning as L
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, LayerNorm, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CNNEncoder(nn.Module):
    """Implementing 1D encoder to extract features from single representation.

    Shape of inputs: (n x 27)

    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Projecting single representation into higher dimension
        self.fc1 = Linear(in_channels, hidden_channels)

        # Feature extraction
        self.conv1 = Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # Decoding layer
        self.conv3 = Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

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


class GCNCoordinatePredictor(nn.Module):
    """GCN based decoder to predict final coordinates of sequence.

    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.bn1 = LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.bn2 = LayerNorm(hidden_channels)
        self.fcn = Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.regress_head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch_size, seq_len):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = x.view(batch_size, seq_len, -1).transpose(1, 2)         # Reshape for CNN
        x = self.fcn(x)
        x = F.leaky_relu(x)
        x = x.transpose(1, 2)

        mask = (x!=-1.0).any(dim=-1)                        # Ensuring padding doesn't disrupt predictions
        x = x * mask.unsqueeze(-1).float()
        coords = self.regress_head(x)

        return coords


class jetRNA_v4_Model(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.cnn_encoder = CNNEncoder(in_channels, hidden_channels, hidden_channels)
        self.gcn_predictor = GCNCoordinatePredictor(hidden_channels, hidden_channels, out_channels)
        self.lr = learning_rate

    def forward(self, x, edges):
        batch_size, seq_len, _ = x.shape
        
        # Extract features from CNN
        features, reconstruction = self.cnn_encoder(x)

        # Flatten for GCN
        x_flat = features.reshape(batch_size * seq_len, -1)

        # Predicting coordinates with GCN
        pred_coords = self.gcn_predictor(x_flat, edges, batch_size, seq_len)

        return pred_coords, reconstruction

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, "val")
        return loss

    def shared_step(self, batch, step):
        x, edges, y, y_center, valid_coords_mask = batch
        print(x.max(), y.max())

        pred_coords, reconstructed = self(x, edges)         # Features are, for now, the predicted coords

        padding_mask = (x!=-1.0).any(dim=-1)                        # Ignoring padding
        mask = padding_mask & valid_coords_mask

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
                sum_sq = torch.sum(diffs*2, dim=-1)
                dists = torch.sqrt(sum_sq.clamp(min=1e-7))

                bond_loss += F.mse_loss(dists, torch.full_like(dists, ideal_dist))

        avg_bond_loss = bond_loss / pred_coords.shape[0]
        total_loss = loss_coord + (0.5 * avg_bond_loss) + (0.1 * loss_rec)

        self.log(f"{step}_loss_rec", loss_rec, batch_size=8)
        self.log(f"{step}_loss_coord", loss_coord, batch_size=8)
        self.log(f"{step}_loss_total", total_loss, batch_size=8)
        return total_loss

    def on_train_start(self):
        #torch.autograd.set_detect_anomaly(True)
        #print("--- Anomaly Detection Enabled ---")
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if torch.isnan(outputs["loss"]):
            print(f"NaN loss detected at batch {batch_idx}")

    def predict_step(self, batch, batch_idx):
        x, edges, y, y_center, mask, id_list = batch
        pred_coords, reconstructed = self(x, edges)         # Features are, for now, the predicted coords

        pred_coords = (pred_coords * 900.0) + y_center.unsqueeze(1)     # Denormalize and recenter

        results = {} 
        for i in range(pred_coords.shape[0]):
            valid_indices = mask[i]
            actual_coords = pred_coords[i][valid_indices]
            results[id_list[i]] = actual_coords.cpu()
             
        return results

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

