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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_svm(x, y, params = None):

    def hp_tuning(param_grid):
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
            svm.fit(x, y)
            accuracy = evaluate_model(svm, x_val, y_val, "validation", False)
            if accuracy >= best_score:
                best_params = params
                best_score = accuracy

        return best_params, best_score
    
    if params == None:
        best_params, best_score = hp_tuning()
        print("Achieved", best_score, "accuracy with params:",best_params)
        svm = SGDClassifier(**best_params)
    else:
        svm = SGDClassifier(**params)

    
    svm.fit(x, y)
    return svm



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





print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)



    

'''x_val, y_val = fetch_data(spect_data, val_keys)

y_val = label_encoder.transform(y_val)
x_val = x_preprocessing.transform(x_val)'''


majority_classifier = Majority_Classifier()
majority_classifier.fit(y_train)
accuracy = evaluate_model(majority_classifier, x_train, y_train, "training")
accuracy = evaluate_model(majority_classifier, x_val, y_val, "validation")

sampling_classifier = Sampling_Classifier()
sampling_classifier.fit(y_train)
accuracy = evaluate_model(sampling_classifier, x_train, y_train, "training")
accuracy = evaluate_model(sampling_classifier, x_val, y_val, "validation")

params = {'alpha': 1000, 'loss': 'log_loss', 'max_iter': 1000, 'random_state': 4}
svm = train_svm(x_train, y_train,params = params )
accuracy = evaluate_model(svm, x_train, y_train, "training")
accuracy = evaluate_model(svm, x_val, y_val, "validation")

ConfusionMatrixDisplay.from_estimator(svm, x_val, y_val)
plt.show()

x_total = np.concatenate([x_train, x_val], axis = 0)
y_total = np.concatenate([y_train, y_val], axis = 0)


svm = train_svm(x_total, y_total,params = params )
accuracy = evaluate_model(svm, x_total, y_total, "training")
accuracy = evaluate_model(svm, x_test, y_test, "test")
ConfusionMatrixDisplay.from_estimator(svm, x_test, y_test)
plt.show()