import h5py
import os 
current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, '..', '..', 'data')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')

spect_data = h5py.File(hdf5_path, 'r')
genre_list = []
for song in spect_data:
    song_data = spect_data.get(song)
    genre_list.append(song_data.attrs['genre'])

genre_set = set(genre_list)
print(genre_set)