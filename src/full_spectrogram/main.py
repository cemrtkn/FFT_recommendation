from config import hdf5_path, spec_minmax_scaler_path, spec_log_transformer_path
from data_loader import SpectLoader
from model import ConvNetwork
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch.nn.functional as F



def data_dist(pprocessor, train_keys, threshold=0.05):
    num_bins = 50
    bin_edges = np.linspace(start=0.0, stop=1.0, num=num_bins + 1)  # Adjust the range based on your data's expected value range

    bin_counts = np.zeros(num_bins)
    for x_batch, _ in pprocessor.batch_generator(train_keys):
        
        data = x_batch.numpy() if hasattr(x_batch, 'numpy') else x_batch

        flattened_data = data.flatten()

        hist, _ = np.histogram(flattened_data, bins=bin_edges)
        bin_counts += hist  

    bin_counts_cumulative = np.cumsum(bin_counts)  

    bin_counts_cumulative = bin_counts_cumulative / bin_counts_cumulative[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[1:], bin_counts_cumulative, color='blue', alpha=0.7, lw=2)
    plt.title("Cumulative Distribution of Values in x_batch")
    plt.xlabel("Value")
    plt.ylabel("Cumulative Frequency")
    plt.grid(True)
    plt.show()

    bin_index_percentile = np.argmax(bin_counts_cumulative >= threshold)
    bin_start = bin_edges[bin_index_percentile]
    bin_end = bin_edges[bin_index_percentile + 1]

    print(f"The bin containing 5% of the data is between {bin_start} and {bin_end}")

    return bin_edges

def plot_spect(sepct):
    plt.figure(figsize=(10, 5))  # Adjust figure size as needed
    plt.imshow(sepct, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')  # Optional: Add a colorbar
    plt.title('Spectrogram Heatmap')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

def plot_several(batch, n_show):
    for i in range(n_show):
        matrix = batch[i].squeeze(0).numpy()
        plot_spect(matrix)


model = ConvNetwork(input_shape=(1, 256, 512))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

paths = {
        "data_path": hdf5_path,
        "scaler_path": spec_minmax_scaler_path
    }

pprocessor = SpectLoader(paths, batch_size=32)
train_keys, val_keys, test_keys = pprocessor.split_data()
#pprocessor.fit_clipper()
pprocessor.setup_pipeline(scaler_type="normalizer",load_model=True)
#print(pprocessor.pipeline)

#_ = data_dist(pprocessor, train_keys)

for x_batch, y_batch in pprocessor.batch_generator(train_keys):
    resized_tensor = F.interpolate(x_batch, size=(256, 512), mode='bilinear', align_corners=False, antialias=True)

    optimizer.zero_grad()    

    outputs = model(resized_tensor)
    loss = criterion(outputs, y_batch)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        _, train_preds = torch.max(outputs, 1)
        train_accuracy = (train_preds == y_batch).float().mean().item()

    print(f'Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}')

