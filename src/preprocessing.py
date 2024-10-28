import h5py
import os 
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import hinge_loss  
from sklearn.decomposition import IncrementalPCA
import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid


## TO:DO SVM BATCHED GRIDSEARCH 
## TO:DO PUT PREPROCESSIN IN A PIPELINE



current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, '..', '..', 'data')
pca_path = os.path.join(extract_directory, 'pca_model.pkl')
pca_path_alt = os.path.join(extract_directory, 'pca_model_alt.pkl')
scaler_path = os.path.join(extract_directory, 'scaler_model.pkl')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')


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

def evaluate_model(model, x, y, mode):

    y_pred = model.predict(x)
    accuracy = accuracy_score(y_pred, y)
    print(f"{mode} accuracy after this batch: {accuracy:.2%}")
    print('-' * 40)
    return accuracy



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
    for batch in batch_generator(spect_data, train_val_keys, 50):
        x_batch, _ = batch
        x_batch = scaler.transform(x_batch)
        pca.partial_fit(x_batch)
        print("batch no.", counter, "was done for pca")
        counter += 1

    joblib.dump(pca, pca_path_alt)




spect_data = h5py.File(hdf5_path, 'r')
data_keys = np.array(list(spect_data.keys()))

genres = []
for song in spect_data:
    song_data = spect_data.get(song)
    genres.append(song_data.attrs['genre'])

unique_genre_labels = np.unique(genres)

label_encoder = LabelEncoder()
label_encoder.fit(unique_genre_labels)
unique_genre_labels = label_encoder.transform(unique_genre_labels)

train_val_keys, test_keys = generate_train_test_indices(data_keys)
train_keys, val_keys = generate_train_test_indices(train_val_keys, test_size=0.1)


#incremental_preprocessors(spect_data, train_val_keys, n_components=50)

scaler = joblib.load(scaler_path)
#pca_30 = joblib.load(pca_path)
pca_50 = joblib.load(pca_path_alt)


x_preprocessing = Pipeline(
    steps=[
        ("scaler",scaler) ,
        ("pca", pca_50),
    ]
)


x_total = np.empty((0,50))
y_total = np.empty((0,))
for batch in batch_generator(spect_data, train_keys, 150):
    x_batch, y_batch = batch

    x_batch = x_preprocessing.transform(x_batch)
    y_batch = label_encoder.transform(y_batch)

    x_total = np.concatenate((x_total, x_batch), axis = 0)
    y_total = np.concatenate((y_total, y_batch), axis = 0)


print(x_total.shape, y_total.shape)

    

x_val, y_val = fetch_data(spect_data, val_keys)

y_val = label_encoder.transform(y_val)
x_val = x_preprocessing.transform(x_val)


# Define hyperparameter grid
param_grid = {
    'alpha': [50, 100, 1000, 10000],
    'loss': ['hinge', 'log_loss'],
    'max_iter': [100, 1000],
    'random_state': list(range(0,20))
}

grid = list(ParameterGrid(param_grid))
best_score = 0
best_params = None

for params in grid:
    #svm = SGDClassifier(loss='log_loss', alpha = 100, max_iter= 1000)
    svm = SGDClassifier(**params)
    svm.fit(x_total, y_total)
    #print('params:', params)
    accuracy = evaluate_model(svm, x_val, y_val, "validation")
    if accuracy >= best_score:
        best_params = params
        best_score = accuracy

print(best_params, best_score)


   