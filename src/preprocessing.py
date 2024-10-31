import numpy as np
import joblib
import h5py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from config import scaler_path, pca_path_alt, all_pprocessed_data_directory
from numpy import savetxt, loadtxt


class DataPreprocessor:
    def __init__(self, data_path, test_size = 0.2, val_size = 0.1 ,n_components=100, batch_size=150):
        self.data_path = data_path
        self.scaler_path = scaler_path
        self.pca_path = pca_path_alt
        self.n_components = n_components
        self.batch_size = batch_size
        self.spect_data = h5py.File(data_path, 'r')
        self.n_samples = len(list(self.spect_data.keys()))
        self.label_encoder = LabelEncoder()
        self.test_size = test_size
        self.val_size = val_size
        self.pipeline = None  # To store the preprocessing pipeline after setup
        self.all_pprocessed_data_path = all_pprocessed_data_directory

    def fetch_data(self, keys):
        x_batch = np.empty((len(keys), 1326634), dtype=np.float64)
        y_batch = np.empty((len(keys),), dtype=object)
        
        for idx, key in enumerate(keys):
            song = self.spect_data.get(key)
            x_batch[idx] = np.array(song['spectrogram']).flatten().reshape(1, -1)
            y_batch[idx] = str(song.attrs['genre'])
        
        return x_batch, y_batch

    def batch_generator(self, keys):
        n_samples = keys.size
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            yield self.fetch_data(keys[start_idx:end_idx])

    def total_from_batches(self, keys):
        x_total, y_total = np.empty((0, self.n_components)), np.empty((0,))
        
        for x_batch, y_batch in self.batch_generator(keys):
            x_batch = self.pipeline.transform(x_batch)
            y_batch = self.label_encoder.transform(y_batch)
            x_total = np.concatenate((x_total, x_batch), axis=0)
            y_total = np.concatenate((y_total, y_batch), axis=0)
        
        return x_total, y_total

    def generate_train_test_indices(self, data_keys, split_mode ,test_size=0.2):
        np.random.seed(42)
        n_samples = len(data_keys)
        np.random.shuffle(data_keys)
        split_idx = int(n_samples * (1 - test_size))

        if split_mode == 'train-test':
            self.train_test_idx = split_idx
        elif split_mode == 'train-val':
            self.train_val_idx = split_idx
        
        return data_keys[:split_idx], data_keys[split_idx:]

    def incremental_preprocessors(self, train_keys, skip_scaler=True):
        if not skip_scaler:
            scaler = StandardScaler()
            for x_batch, _ in self.batch_generator(train_keys):
                scaler.partial_fit(x_batch)
            joblib.dump(scaler, self.scaler_path)
        else:
            scaler = joblib.load(self.scaler_path)

        pca = IncrementalPCA(n_components=self.n_components)
        for x_batch, _ in self.batch_generator(train_keys):
            x_batch = scaler.transform(x_batch)
            pca.partial_fit(x_batch)
        joblib.dump(pca, self.pca_path)

    def setup_pipeline(self):
        scaler = joblib.load(self.scaler_path)
        pca = joblib.load(self.pca_path)
        self.pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca)])

    def prepare_data(self):
        data_keys = np.array(list(self.spect_data.keys()))
        genres = [self.spect_data[key].attrs['genre'] for key in self.spect_data]
        
        unique_genre_labels = np.unique(genres)
        self.label_encoder.fit(unique_genre_labels)

        train_val_keys, test_keys = self.generate_train_test_indices(data_keys, 'train-test')
        train_keys, val_keys = self.generate_train_test_indices(train_val_keys, 'train-val',test_size=0.1)

        self.setup_pipeline()
        
        x_train, y_train = self.total_from_batches(train_keys)
        x_val, y_val = self.total_from_batches(val_keys)
        x_test, y_test = self.total_from_batches(test_keys)

        all_pprocessed_x = np.concatenate([x_train, x_val, x_test], axis = 0)
        all_pprocessed_y = np.concatenate([y_train, y_val, y_test], axis = 0)
        all_pprocessed_data = np.concatenate([all_pprocessed_x, all_pprocessed_y[:, np.newaxis]], axis=1)


        savetxt(self.all_pprocessed_data_path, all_pprocessed_data, delimiter=',')
        #savetxt(self.all_pprocessed_y_path, all_pprocessed_y, delimiter=',')

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
        }
    def load_pprocessed_data(self):
        all_pprocessed_data = loadtxt(self.all_pprocessed_data_path, delimiter=',')

        all_pprocessed_x = all_pprocessed_data[:,:self.n_components]
        all_pprocessed_y = all_pprocessed_data[:,-1]

        train_test_split_idx = int(self.n_samples * (1 - self.test_size))
        train_val_split_idx = int(train_test_split_idx * (1 - self.val_size))

        return {
            "x_train": all_pprocessed_x[:train_val_split_idx],
            "y_train": all_pprocessed_y[:train_val_split_idx],
            "x_val": all_pprocessed_x[train_val_split_idx:train_test_split_idx],
            "y_val": all_pprocessed_y[train_val_split_idx:train_test_split_idx],
            "x_test": all_pprocessed_x[train_test_split_idx:],
            "y_test": all_pprocessed_y[train_test_split_idx:],
        }


