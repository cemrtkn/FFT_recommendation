from config import hdf5_path, spec_scaler_path
from data_loader import SpectLoader
from model import ConvNetwork
import torch
from torch import nn, optim

model = ConvNetwork(input_shape=(1, 257, 5162))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.00001)

paths = {
        "data_path": hdf5_path,
        "scaler_path": spec_scaler_path
    }

pprocessor = SpectLoader(paths, batch_size=32)
train_keys, val_keys, test_keys = pprocessor.split_data()
pprocessor.setup_pipeline(load_model=False)

for x_batch, y_batch in pprocessor.batch_generator(train_keys):
    print(x_batch[0].dtype, len(y_batch))
    print(x_batch[0].shape)
    optimizer.zero_grad()

    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        _, train_preds = torch.max(outputs, 1)
        train_accuracy = (train_preds == y_batch).float().mean().item()
    
    print(f'Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}')

