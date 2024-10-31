import os

current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, '..', '..', 'data')
pca_path = os.path.join(extract_directory, 'pca_model.pkl')
pca_path_alt = os.path.join(extract_directory, 'pca_model_alt.pkl')
scaler_path = os.path.join(extract_directory, 'scaler_model.pkl')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')
file_path = os.path.join(current_directory, '..', '..', 'data', 'fma_small.zip')
tracks_extract_directory = os.path.join(current_directory, '..', '..', 'data', 'tracks.csv')
all_pprocessed_data_directory = os.path.join(extract_directory, 'all_pprocessed_data.csv')


