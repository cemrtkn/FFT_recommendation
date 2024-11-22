import os

current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, ".." ,'..', '..', 'data')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')
spec_minmax_scaler_path = os.path.join(extract_directory, 'spec_normalizer_path.h5')
spec_log_transformer_path = os.path.join(extract_directory, 'spec_log_transformer_path.h5')





