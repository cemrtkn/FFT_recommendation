import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import hdf5_path, spec_minmax_scaler_path
import torch
from sklearn.pipeline import Pipeline
import joblib
from custom_preproc import MinMaxScaler, LogTransformer


# TO:DO don't split data before crop augmentation
np.random.seed(42)

scalers = {
    "normalizer": MinMaxScaler(),
    "log_transformer": LogTransformer(),
}

class SpectLoader:
    def __init__(self, paths, test_size = 0.2, val_size = 0.1, batch_size=64):
        self.data_path = paths['data_path']
        self.scaler_path = paths['scaler_path']

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
        self.train_keys = None
        self.max_db = None
        self.min_db = None

    
    def split_data(self,):
        n_samples = len(self.data_keys)
        np.random.shuffle(self.data_keys)

        train_val_idx = int(n_samples * (1 - (self.val_size + self.test_size)))
        train_test_idx = int(n_samples * (1 - self.test_size))

        train_keys = self.data_keys[:train_val_idx]
        self.train_keys = train_keys
        val_keys = self.data_keys[train_val_idx:train_test_idx]
        test_keys = self.data_keys[train_test_idx:]

        return train_keys, val_keys, test_keys

    def fetch_data(self, batch_keys, fitting_scaler=False):
        x_batch = np.empty((len(batch_keys), self.n_freq, self.n_time), dtype=np.float32)
        y_batch = np.empty((len(batch_keys),), dtype=object)
        
        for idx, key in enumerate(batch_keys):
            song = self.spect_data.get(key)
            x_batch[idx] = np.array(song['spectrogram'])
            y_batch[idx] = song.attrs['genre']
        
        if not fitting_scaler:
            x_batch = self.pipeline['scaler'].transform(x_batch)
            y_batch = self.pipeline['target_encoder'].transform(y_batch)
        else:
            y_batch = np.empty((len(batch_keys),), dtype=np.int32)

        # Convert x_batch to torch tensor and add channel dimension
        x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device).unsqueeze(1)  # Add channel dimension
        y_batch = torch.tensor(y_batch, dtype=torch.long, device=self.device)
        return x_batch, y_batch

    def batch_generator(self, split_keys, fitting_scaler = False):
        split_size = len(split_keys)
        for start_idx in range(0, split_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, split_size)
            yield self.fetch_data(split_keys[start_idx:end_idx], fitting_scaler=fitting_scaler)
    
    def setup_pipeline(self, scaler_type = "normalizer" ,load_model=True):
        scaler = scalers[scaler_type]

        genres = [self.spect_data[key].attrs['genre'] for key in self.spect_data]
        unique_genre_labels = np.unique(genres)
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_genre_labels)

        if load_model:
            custom_scaler = joblib.load(self.scaler_path)
            print(custom_scaler.__dict__)
            scaler.set_params(**custom_scaler.__dict__)
        else: 
            counter = 0
            # incrementally train scaler
            for x_batch, _ in self.batch_generator(self.train_keys, fitting_scaler=True):
                scaler.partial_fit(x_batch)
                print("Iteration no:", counter, "is over.")
                counter += 1
            joblib.dump(scaler, self.scaler_path)
        
        self.pipeline = Pipeline([
                ("scaler", scaler),
            ("target_encoder", label_encoder),
        ])
    
    def set_min_max(self):
        max = 0
        min = 0
        for x_batch, _ in self.batch_generator(self.train_keys):
            batch_max = torch.max(x_batch).item()
            batch_min = torch.min(x_batch).item()
            if batch_max > max:
                max = batch_max
            if batch_min < min:
                min = batch_min
        
        self.max_db = max
        self.min_db = min

    def normalize(self):
        assert self.max_db is not None

        




if __name__ == "__main__":
    paths = {
        "data_path": hdf5_path,
        "scaler_path": spec_scaler_path
    }
    pprocessor = SpectLoader(paths)
    train_keys, val_keys, test_keys = pprocessor.split_data()
    pprocessor.setup_pipeline(scaler_type="log_transformer",load_model=False)
    

