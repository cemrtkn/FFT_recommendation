import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import joblib
from config import *

def total_from_batches(spect_data, keys, x_preprocessing, label_encoder):
    x_total = np.empty((0,100))
    y_total = np.empty((0,))

    for batch in batch_generator(spect_data,keys , 150):
        x_batch, y_batch = batch

        x_batch = x_preprocessing.transform(x_batch)
        y_batch = label_encoder.transform(y_batch)

        x_total = np.concatenate((x_total, x_batch), axis = 0)
        y_total = np.concatenate((y_total, y_batch), axis = 0)
    return x_total, y_total


def generate_train_test_indices(data_keys, test_size=0.2):

    np.random.seed(42)

    n_samples = len(data_keys)
    np.random.shuffle(data_keys)
    split_idx = int(n_samples * (1 - test_size))

    train_keys = data_keys[:split_idx]
    test_keys = data_keys[split_idx:]

    
    
    return train_keys, test_keys

def fetch_data(data, keys):
    
    x_batch = np.empty((len(keys), 1326634), dtype=np.float64) 
    y_batch = np.empty((len(keys),), dtype=object)
    for idx, key in enumerate(keys):

        song = data.get(key)
        new_x_element = np.array(song['spectrogram']).flatten().reshape(1, -1)
        new_y_element = str(song.attrs['genre'])      

        x_batch[idx] = new_x_element
        y_batch[idx] = new_y_element
    return x_batch, y_batch

def batch_generator(data, keys, batch_size=150):
    n_samples = keys.size
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        curr_batch_keys = keys[start_idx:end_idx]

        x_batch, y_batch = fetch_data(data, curr_batch_keys)

        yield x_batch, y_batch



def incremental_preprocessors(spect_data, train_val_keys, skip_scaler = True , n_components = 30):
    if not skip_scaler:
        scaler = StandardScaler()

        counter = 1
        for batch in batch_generator(spect_data, train_val_keys):
            x_batch, _ = batch
            scaler.partial_fit(x_batch)
            print("batch no.", counter, "was done for the scaler")
            counter += 1

        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    pca = IncrementalPCA(n_components=n_components)

    counter = 1
    for batch in batch_generator(spect_data, train_val_keys, 100):
        x_batch, _ = batch
        x_batch = scaler.transform(x_batch)
        pca.partial_fit(x_batch)
        print("batch no.", counter, "was done for pca")
        counter += 1

    joblib.dump(pca, pca_path_alt)







   