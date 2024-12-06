import os

current_directory = os.getcwd()
extract_directory = os.path.join(current_directory,'..', '..', 'data')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')
hdf5_path_toy = os.path.join(extract_directory, 'spectrograms_toy.h5')
dataset_path = os.path.join(extract_directory, 'fma_small.zip')
tracks_metadata_path = os.path.join(extract_directory, 'tracks.csv')







