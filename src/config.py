import os

current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, ".." ,'..', '..', 'data')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')
spec_scaler_path = os.path.join(extract_directory, 'spec_scaler_path.h5')




