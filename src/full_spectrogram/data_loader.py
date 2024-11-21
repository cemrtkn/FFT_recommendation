import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import hdf5_path
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



# TO:DO don't split data before crop augmentation
np.random.seed(42)

class SpectLoader:
    def __init__(self, paths, test_size = 0.2, val_size = 0.1, batch_size=64):
        self.data_path = paths['data_path']
        #self.scaler_path = paths['scaler_path']

        self.batch_size = batch_size
        self.spect_data = h5py.File(self.data_path, 'r')
        self.data_keys = list(self.spect_data.keys())
        self.n_samples = len(self.data_keys)
        self.n_freq = self.spect_data.get(self.data_keys[0])['spectrogram'].shape[0]
        self.n_time = self.spect_data.get(self.data_keys[0])['spectrogram'].shape[1]
        self.test_size = test_size
        self.val_size = (1-test_size)*val_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pipeline = None

        self.spect_data
    
    def split_data(self,):
        n_samples = len(self.data_keys)
        np.random.shuffle(self.data_keys)

        train_val_idx = int(n_samples * (1 - (self.val_size + self.test_size)))
        train_test_idx = int(n_samples * (1 - self.test_size))

        train_keys = self.data_keys[:train_val_idx]
        val_keys = self.data_keys[train_val_idx:train_test_idx]
        test_keys = self.data_keys[train_test_idx:]

        return train_keys, val_keys, test_keys

    def fetch_data(self, batch_keys):
        assert self.pipeline != None
        x_batch = np.empty((len(batch_keys), self.n_freq, self.n_time), dtype=np.float64)
        y_batch = np.empty((len(batch_keys),), dtype=object)
        
        for idx, key in enumerate(batch_keys):
            song = self.spect_data.get(key)
            x_batch[idx] = np.array(song['spectrogram'])
            y_batch[idx] = str(song.attrs['genre'])
        
        y_batch = self.pipeline['target_encoder'].transform(y_batch)

        x_batch = torch.tensor(x_batch, dtype=torch.float64, device=self.device)  
        y_batch = torch.tensor(y_batch, dtype=torch.long, device=self.device)
        return x_batch, y_batch

    def batch_generator(self, split_keys):
        split_size = len(split_keys)
        for start_idx in range(0, split_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, split_size)
            yield self.fetch_data(split_keys[start_idx:end_idx])
    
    def setup_pipeline(self):
        self.pipeline = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("scaler", StandardScaler()),  # Transform X
            ])),
            ("target_encoder", LabelEncoder()),
        ])
        genres = [self.spect_data[key].attrs['genre'] for key in self.spect_data]
        unique_genre_labels = np.unique(genres)
        self.pipeline['target_encoder'].fit(unique_genre_labels)
    




if __name__ == "__main__":
    paths = {
        "data_path": hdf5_path
    }
    pprocessor = SpectLoader(paths)
    pprocessor.setup_pipeline()
    train_keys, val_keys, test_keys = pprocessor.split_data()

    for x_batch, y_batch in pprocessor.batch_generator(train_keys):
        print(len(x_batch), len(y_batch))
        print(x_batch[0].shape)
        break

