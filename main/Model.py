import torch
import lightning as L
import torch.nn as nn
from torch.nn import Conv1d, Dropout, Identity, LayerNorm, LeakyReLU, Linear, BatchNorm1d, ModuleList, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CNNResBlock(nn.Module):
    """Residual block for repeatability.

    """
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        # Residual block 1
        self.conv1 = Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(hidden_channels)

        # Residual block 2
        self.conv2 = Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(hidden_channels)

        self.drop = Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        features = F.leaky_relu(out + identity)
        return features 


class CNNEncoder(nn.Module):
    """Implementing 1D CNN encoder to extract features from single representation.

    Shape of inputs: (n x 27)

    """
    def __init__(self, in_channels, hidden_channels, num_blocks, dropout):
        super().__init__()
        # Projecting single representation into higher dimension
        self.initial_projection = Linear(in_channels, hidden_channels)
        self.ln1 = LayerNorm(hidden_channels)

        # Feature extraction
        self.blocks = ModuleList([
            CNNResBlock(hidden_channels, dropout) for _ in range(num_blocks)
        ])

        # Decoding layer
        self.reconstruction = Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_projection(x)
        x = self.ln1(x)
        x = F.leaky_relu(x)

        # Transpose to (batch, 27, n)
        x = x.transpose(1, 2)

        # Residual block
        for block in self.blocks:
            x = block(x)

        features = x
        reconstruction = self.reconstruction(features)

        # Transpose back to (batch, n, hidden_channels)
        features= features.transpose(1, 2)
        reconstruction = reconstruction.transpose(1, 2)

        return features, reconstruction 


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=True, normalize=True)
        self.ln = LayerNorm(out_channels)
        self.dropout = Dropout(dropout)

        self.shortcut = Linear(in_channels, out_channels) if in_channels != out_channels else Identity()

    def forward(self, x, edge_index):
        identity = self.shortcut(x)
        out = self.conv(x, edge_index)
        out = self.ln(out)
        out = F.leaky_relu(out)
        out = self.dropout(out)
        out = out + identity            # Residual connection
        return out


class GCNCoordinatePredictor(nn.Module):
    """GCN based decoder to predict final coordinates of sequence.

    """
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        # GCN stack
        layer_configs = [
            (in_channels, 64),
            (64, 128),
            (128, 256),
            (256, 256)
        ]
        self.gcn_layers = ModuleList([
            GCNBlock(i, o, dropout) for i, o in layer_configs
        ])

        # Local refinement
        self.refine_cnn = Conv1d(256, 128, kernel_size=5, padding=2)
        self.refine_ln = LayerNorm(128)

        # Regression head using MLP
        self.regress_head = Sequential(
            Linear(128, 64),
            LeakyReLU(0.01),
            Dropout(dropout),
            Linear(64, out_channels)
        )

    def forward(self, x, edge_index, batch_size, seq_len):
        for layer in self.gcn_layers:
            x = layer(x, edge_index)

        # Reshape for CNN 
        x = x.view(batch_size, seq_len, -1).transpose(1, 2)         # Reshape for CNN
        x = self.refine_cnn(x)
        x = F.leaky_relu(x)
        x = x.transpose(1, 2)
        x = self.refine_ln(x)

        # Prediction
        #mask = (x!=-1.0).any(dim=-1)                        # Ensuring padding doesn't disrupt predictions
        #x = x * mask.unsqueeze(-1).float()
        mask = (x.abs() > 1e-8).any(dim=-1, keepdim=True)
        x = x * mask.float()
        coords = self.regress_head(x)

        return coords


class jetRNA_v4_Model(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, dropout, num_cnn_blocks):
        super().__init__()
        self.save_hyperparameters()
        self.cnn_encoder = CNNEncoder(in_channels, hidden_channels, num_cnn_blocks, dropout)
        self.gcn_predictor = GCNCoordinatePredictor(hidden_channels, out_channels, dropout)
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
        x, edges, y, y_center, valid_coords_mask, target_id = batch
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
                sum_sq = torch.sum(diffs**2, dim=-1)
                dists = torch.sqrt(sum_sq.clamp(min=1e-7))

                bond_loss += F.mse_loss(dists, torch.full_like(dists, ideal_dist))

        # Calculating RMSD (distance in Angstroms)
        rmsd = 900.0 * torch.sqrt(loss_coord.detach().clamp(min=1e-9))      # .detach() so this metric doesn't track gradients; .clamp(min=1e-9) to avoid sqrt(0) which can cause NaNs

        avg_bond_loss = bond_loss / pred_coords.shape[0]
        total_loss = loss_coord + (0.5 * avg_bond_loss) + (0.1 * loss_rec)

        # Loss will be scaled
        bs = x.shape[0]
        self.log(f"{step}_loss_rec", loss_rec, batch_size=bs)
        self.log(f"{step}_loss_coord", loss_coord, batch_size=bs)
        self.log(f"{step}_loss_total", total_loss, batch_size=bs, prog_bar=True)
        self.log(f"{step}_rmsd_angstroms", rmsd, batch_size=bs, prog_bar=True)
        return total_loss

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

