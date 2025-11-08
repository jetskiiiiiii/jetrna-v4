import torch
import lightning as L
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from jetRNADataModule import jetRNADataModule

class CoordinatePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.regress_head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.regress_head(x)
        return x

class jetRNAGNNModule(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.save_hyperparameters()
        self.model = CoordinatePredictor(in_channels, hidden_channels, out_channels)
        self.lr = 0.01

    def forward(self, data: Data):
        return self.model(data.x, data.edge_index)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log('train_loss', loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log('val_loss', loss, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == '__main__':
    # Hyperparameters
    IN_DIM = 5
    HIDDEN_DIM = 5
    OUT_DIM = 3
    LEARNING_RATE = 0.01
    BATCH_SIZE = 2 # Batches 2 variable-sized graphs together

    # Instantiate modules
    model = jetRNAGNNModule(IN_DIM, HIDDEN_DIM, OUT_DIM)
    data_module = jetRNADataModule()

    # Initialize a Trainer
    trainer = L.Trainer(
        max_epochs=5,
        log_every_n_steps=1,
        enable_checkpointing=True, # Disable complex features for simplicity
        accelerator="mps"
    )

    print("\n--- Starting Training (5 Epochs) ---")
    trainer.fit(model, data_module)

    print("\n--- Training Complete ---")
    print("Final validation loss should appear in the output log.")

    model.eval() 
    with torch.no_grad():
        prediction = trainer.predict(datamodule=data_module)
        print(prediction[2])
