import numpy as np
import joblib
import h5py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from config import hdf5_path, pca_path_aug, scaler_path_aug, all_pprocessed_data_directory_aug
from numpy import savetxt, loadtxt

# TO:DO don't split data before crop augmentation
np.random.seed(42)

class DataPreprocessor:
    def __init__(self, paths, crop_augment_fold, test_size = 0.2, val_size = 0.1 ,n_components=100, batch_size=120):
        self.data_path = paths['data_path']
        self.scaler_path = paths['scaler_path']
        self.pca_path = paths['pca_path']
        self.all_pprocessed_data_path = paths['all_pprocessed_data_path']

        self.n_components = n_components
        self.batch_size = batch_size
        self.spect_data = h5py.File(self.data_path, 'r')
        self.n_samples = len(list(self.spect_data.keys()))*crop_augment_fold
        self.label_encoder = LabelEncoder()
        self.test_size = test_size
        self.val_size = val_size
        self.pipeline = None  # To store the preprocessing pipeline after setup
        self.flat_spec_len = np.array(self.spect_data.get(list(self.spect_data.keys())[0])['spectrogram']).flatten().shape[0]
        self.crop_augment_fold = crop_augment_fold


    def fetch_data(self, keys):
        # make it divisible for augmentation
        divisible_len = self.flat_spec_len - (self.flat_spec_len % self.crop_augment_fold)
        x_batch = np.empty((len(keys), divisible_len), dtype=np.float64)
        y_batch = np.empty((len(keys),), dtype=object)
        
        for idx, key in enumerate(keys):
            song = self.spect_data.get(key)
            x_batch[idx] = np.array(song['spectrogram']).flatten('F').reshape(1, -1)[:, :divisible_len] 
            y_batch[idx] = str(song.attrs['genre'])
        
        print("Fetched data with length", divisible_len)
        return x_batch, y_batch

    def batch_generator(self, keys):
        n_samples = keys.size
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            yield self.fetch_data(keys[start_idx:end_idx])
    
    def data_augmentation(self, x_batch, y_batch):
        num_cols = x_batch.shape[1]
        num_rows = x_batch.shape[0]

        x_batch = np.reshape(x_batch, (int(num_rows*3), int(num_cols/3)))
        y_batch = np.repeat(y_batch, 3)

        return x_batch, y_batch

    def total_from_batches(self, keys, augmentation = False):
        x_total, y_total = np.empty((0, self.n_components)), np.empty((0,))
        
        for x_batch, y_batch in self.batch_generator(keys):
            if augmentation:
                x_batch, y_batch = self.data_augmentation(x_batch, y_batch)
            x_batch = self.pipeline.transform(x_batch)
            y_batch = self.label_encoder.transform(y_batch)
            x_total = np.concatenate((x_total, x_batch), axis=0)
            y_total = np.concatenate((y_total, y_batch), axis=0)
        
        return x_total, y_total

    def generate_train_test_indices(self, data_keys, split_mode ,test_size=0.2):
        n_samples = len(data_keys)
        np.random.shuffle(data_keys)
        split_idx = int(n_samples * (1 - test_size))

        if split_mode == 'train-test':
            self.train_test_idx = split_idx
        elif split_mode == 'train-val':
            self.train_val_idx = split_idx

        return data_keys[:split_idx], data_keys[split_idx:]

    def incremental_preprocessors(self, train_keys, augmentation=False, skip_scaler=True):
        if not skip_scaler:
            scaler = StandardScaler()
            for x_batch, _ in self.batch_generator(train_keys):
                if augmentation:
                    x_batch, _ = self.data_augmentation(x_batch, _)
                scaler.partial_fit(x_batch)
            joblib.dump(scaler, self.scaler_path)
            print("Trained scaler and saved at",self.scaler_path)
        else:
            scaler = joblib.load(self.scaler_path)
            print("Loaded scaler")
        pca = IncrementalPCA(n_components=self.n_components)
        counter = 1
        print((self.n_samples/self.batch_size), "expected for pca training.")
        for x_batch, _ in self.batch_generator(train_keys):
            if augmentation:
                x_batch, _ = self.data_augmentation(x_batch, _)
            x_batch = scaler.transform(x_batch)
            pca.partial_fit(x_batch)
            print("Iteration no", counter, "was done")
            counter += 1
            
        joblib.dump(pca, self.pca_path)
        print("Trained pca and saved at", self.pca_path)
    
    def setup_pipeline(self):
        scaler = joblib.load(self.scaler_path)
        pca = joblib.load(self.pca_path)
        self.pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca)])
    
    def unison_shuffled_copies(self,x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def prepare_data(self, augmentation = False, retrain_pprocessors = False):
        data_keys = np.array(list(self.spect_data.keys()))
        genres = [self.spect_data[key].attrs['genre'] for key in self.spect_data]
        
        unique_genre_labels = np.unique(genres)
        self.label_encoder.fit(unique_genre_labels)

        train_val_keys, test_keys = self.generate_train_test_indices(data_keys, 'train-test')
        train_keys, val_keys = self.generate_train_test_indices(train_val_keys, 'train-val',test_size=0.1)

        if augmentation:
            self.n_samples = len(list(self.spect_data.keys()))*self.crop_augment_fold
        if retrain_pprocessors:
            self.incremental_preprocessors(train_keys, augmentation=augmentation, skip_scaler=False)

        self.setup_pipeline()
        
        x_train, y_train = self.total_from_batches(train_keys, augmentation=augmentation)
        x_val, y_val = self.total_from_batches(val_keys, augmentation=augmentation)
        x_test, y_test = self.total_from_batches(test_keys, augmentation=augmentation)
        print(x_train.shape, y_train.shape)

        # shuffle after crop augmentation
        if augmentation:
            x_train, y_train = self.unison_shuffled_copies(x_train, y_train)
            x_val, y_val = self.unison_shuffled_copies(x_val, y_val)
            x_test, y_test = self.unison_shuffled_copies(x_test, y_test)

        all_pprocessed_x = np.concatenate([x_train, x_val, x_test], axis = 0)
        all_pprocessed_y = np.concatenate([y_train, y_val, y_test], axis = 0)
        all_pprocessed_data = np.concatenate([all_pprocessed_x, all_pprocessed_y[:, np.newaxis]], axis=1)


        savetxt(self.all_pprocessed_data_path, all_pprocessed_data, delimiter=',')
        print("Saved all data at",self.all_pprocessed_data_path,"with shape", all_pprocessed_data.shape )
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

        # TO:DO figure out why 2 has to be subtracted to match sizes and get rid
        train_test_split_idx = int(self.n_samples * (1 - self.test_size))-2
        train_val_split_idx = int(train_test_split_idx * (1 - self.val_size))

        return {
            "x_train": all_pprocessed_x[:train_val_split_idx],
            "y_train": all_pprocessed_y[:train_val_split_idx],
            "x_val": all_pprocessed_x[train_val_split_idx:train_test_split_idx],
            "y_val": all_pprocessed_y[train_val_split_idx:train_test_split_idx],
            "x_test": all_pprocessed_x[train_test_split_idx:],
            "y_test": all_pprocessed_y[train_test_split_idx:],
        }


if __name__ == "__main__":
    paths = {
        'data_path': hdf5_path,
        'scaler_path':scaler_path_aug,
        'pca_path':pca_path_aug,
        'all_pprocessed_data_path':all_pprocessed_data_directory_aug,
    }
    # decrease batch size to prevent memory overload while training pca
    data_preprocessor = DataPreprocessor(paths, crop_augment_fold = 3 ,batch_size=120)

    data_splits = data_preprocessor.prepare_data(augmentation=False, retrain_pprocessors=True)

    x_train, y_train, x_val, y_val, x_test, y_test = data_splits['x_train'], data_splits['y_train'], data_splits['x_val'], data_splits['y_val'], data_splits['x_test'], data_splits['y_test']
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)




