import numpy as np
from config import hdf5_path
from preprocessing import DataPreprocessor
from model_utils import *
import torch
import torch.nn as nn
import torch.optim as optim


data_preprocessor = DataPreprocessor(hdf5_path)
#data_splits = data_preprocessor.prepare_data()
data_splits = data_preprocessor.load_pprocessed_data()

x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


model = nn.Sequential(
    nn.Linear(100, 512),
    nn.Dropout(p=0.2),
    nn.ReLU(),
    nn.Linear(512, 8),
)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Set device for training (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# Training parameters
num_epochs = 300
batch_size = 64

# Training loop
for epoch in range(num_epochs):
    model.train()
    # Shuffle the training data
    permutation = torch.randperm(x_train_tensor.size(0))

    for i in range(0, x_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train_tensor[indices], y_train_tensor[indices]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        _, val_preds = torch.max(val_outputs, 1)
        val_accuracy = (val_preds == y_val_tensor).float().mean().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')
