from config import hdf5_path, spec_minmax_scaler_path, spec_log_transformer_path
from data_loader import SpectLoader
from model import ConvNetwork
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

model = ConvNetwork(input_shape=(1, 257, 5162))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.00001)

paths = {
        "data_path": hdf5_path,
        "scaler_path": spec_minmax_scaler_path
    }

pprocessor = SpectLoader(paths, batch_size=32)
train_keys, val_keys, test_keys = pprocessor.split_data()
pprocessor.setup_pipeline(scaler_type="normalizer",load_model=True)
print(pprocessor.pipeline)

num_bins = 50
bin_edges = np.linspace(start=0.0, stop=1.0, num=num_bins + 1)  # Adjust the range based on your data's expected value range

bin_counts = np.zeros(num_bins)

for x_batch, y_batch in pprocessor.batch_generator(train_keys):
    
    data = x_batch.numpy() if hasattr(x_batch, 'numpy') else x_batch

    flattened_data = data.flatten()

    hist, _ = np.histogram(flattened_data, bins=bin_edges)
    bin_counts += hist  # Add to the cumulative bin counts

bin_counts_cumulative = np.cumsum(bin_counts)  

bin_counts_cumulative = bin_counts_cumulative / bin_counts_cumulative[-1]

print(bin_counts_cumulative)

plt.figure(figsize=(10, 6))
plt.plot(bin_edges[1:], bin_counts_cumulative, color='blue', alpha=0.7, lw=2)
plt.title("Cumulative Distribution of Values in x_batch")
plt.xlabel("Value")
plt.ylabel("Cumulative Frequency")
plt.grid(True)

# Show the plot
plt.show()

"""optimizer.zero_grad()

outputs = model(x_batch)
loss = criterion(outputs, y_batch)

loss.backward()
optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    _, train_preds = torch.max(outputs, 1)
    train_accuracy = (train_preds == y_batch).float().mean().item()

print(f'Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}')"""

