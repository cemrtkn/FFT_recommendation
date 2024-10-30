import numpy as np
from config import *
from preprocessing import *
from sklearn.preprocessing import LabelEncoder
import h5py
from sklearn.pipeline import Pipeline
from model_utils import *

# TO:DO fix the repetition that happened with this 
# script and sgd_training for data data preprocessing

spect_data = h5py.File(hdf5_path, 'r')
data_keys = np.array(list(spect_data.keys()))

genres = []
for song in spect_data:
    song_data = spect_data.get(song)
    genres.append(song_data.attrs['genre'])

unique_genre_labels = np.unique(genres)
print(unique_genre_labels)

label_encoder = LabelEncoder()
label_encoder.fit(unique_genre_labels)
unique_genre_labels = label_encoder.transform(unique_genre_labels)

train_val_keys, test_keys = generate_train_test_indices(data_keys)
train_keys, val_keys = generate_train_test_indices(train_val_keys, test_size=0.1)


#incremental_preprocessors(spect_data, train_val_keys, n_components=100)

scaler = joblib.load(scaler_path)
#pca_30 = joblib.load(pca_path)
pca_100 = joblib.load(pca_path_alt)


x_preprocessing = Pipeline(
    steps=[
        ("scaler",scaler) ,
        ("pca", pca_100),
    ]
)


x_train, y_train = total_from_batches(spect_data, train_keys, x_preprocessing, label_encoder)
x_val, y_val = total_from_batches(spect_data, val_keys, x_preprocessing, label_encoder)
x_test, y_test = total_from_batches(spect_data, test_keys, x_preprocessing, label_encoder)

print("going into training")


param_grid = {
    'eta': np.arange(0.01, 1, 0.05),
    'alpha': np.arange(0, 10, 1),
}

#train_xgb(x_train, y_train, 'xgb', x_val, y_val, param_grid = param_grid)

best_params = {'eta': 0.51, 'alpha': 6}
model = train_model(x_train, y_train,'xgb', params = best_params)
evaluate_model(model, x_train, y_train, "training")
evaluate_model(model, x_val, y_val, "validation")


