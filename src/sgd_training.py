import h5py
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from preprocessing import *
from dummy_classifiers import *
from config import *

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


majority_classifier = Majority_Classifier()
majority_classifier.fit(y_total)
accuracy = evaluate_model(majority_classifier, x_total, y_total, "training")
accuracy = evaluate_model(majority_classifier, x_val, y_val, "validation")

sampling_classifier = Sampling_Classifier()
sampling_classifier.fit(y_total)
accuracy = evaluate_model(sampling_classifier, x_total, y_total, "training")
accuracy = evaluate_model(sampling_classifier, x_val, y_val, "validation")

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
    svm = SGDClassifier(**params)
    svm.fit(x_total, y_total)
    accuracy = evaluate_model(svm, x_val, y_val, "validation", False)
    if accuracy >= best_score:
        best_params = params
        best_score = accuracy

print(best_params, best_score)
svm = SGDClassifier(**best_params)
svm.fit(x_total, y_total)
accuracy = evaluate_model(svm, x_total, y_total, "training")
accuracy = evaluate_model(svm, x_val, y_val, "validation")