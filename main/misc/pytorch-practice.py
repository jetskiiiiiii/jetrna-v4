import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

device = None

# Check if using Apple Sillicon MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"PyTorch is using MPS")
else:
    print("MPS is not available on this device.")


## compute per-channel mean and standard deviation for a batch of images (commonly used in normalization)
# Simulate a batch of 64 images, 32x32
data = torch.rand((64, 3, 32, 32))

# Calculate mean and std for each channel
mean = data.mean(dim=(0, 2, 3))
std = data.std(dim=(0, 2, 3))

# Normalize batch
normalized_data = (data - mean[None, :, None, None]) / std[None, :, None, None]
print("Normalized Batch Shape:", normalized_data.shape)


## MPS boost
# Move data to GPU
tensor = torch.randn(1000, 1000).to(device)

# Perform a matrix multiplication
result = torch.mm(tensor, tensor.T)

print("Result computed on:", device)


## Creating a custom model architecture
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

model = CustomModel(10, 50, 1)  # Example: 10 input features, 50 hidden nodes, 1 output
print(model)


## adjusts weights to account for the activation’s properties with Kaiming Initialization
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)

model.apply(initialize_weights)


## Creating a Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


data = torch.randn(1000, 10)
labels = torch.randint(0, 2, (1000,))
dataset = CustomDataset(data, labels)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True) # Running num_workers > 0 causes error on MacOS
for batch_data, batch_labels in dataloader:
    print("Batch data shape: ", batch_data.shape)
    print("Batch labels shape: ", batch_labels.shape)


# Training loop
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train() # Set model to training mode
    for inputs, targets in dataloader:
        optimizer.zero_grad() # Clear gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step() # Update weights


## Evaluating model
model.eval()
correct = 0
total = 0

with torch.no_grad(): # Disable gradient calculation for faster computation
    for inputs, targets in dataloader:
        outputs = model(inputs) # Forward pass
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")


## Tips
# call this right after loss.backward() in your training loop
# ensures your gradients stay within a reasonable range
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping

# Diagnosing NaNs in loss
if torch.isnan(loss):
    print("NaN detected in loss! Investigate your input data or model initialization.")

# ProfilerActivity.CPU: Captures CPU performance.
# ProfilerActivity.CUDA: Tracks GPU execution time.
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA],
    on_trace_ready=profiler.tensorboard_trace_handler('./log')
) as prof:
    outputs = model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
