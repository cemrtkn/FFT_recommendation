import zipfile
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TCON
import json
import os
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import h5py




SAMPLING_FREQ = 44100
FIGURE_WIDTH = 7 
FIGURE_HEIGHT = 3.5  
MIN_SAMPLE_LENGTH = 1321967

# Get the current working directory
current_directory = os.getcwd()
file_path = os.path.join(current_directory, '..', '..', 'data', 'fma_small.zip')
genre_mapping_directory = os.path.join(current_directory, '..', '..', 'data', 'parent_mapping.json')
extract_directory = os.path.join(current_directory, '..', '..', 'data')
hdf5_path = os.path.join(extract_directory, 'spectrograms.h5')


def visualize_audio(mono_signal):
    # Get time domain representation of the sound pressure waves
    timeArray = np.arange(0, mono_signal.shape[0], 1.0)
    timeArray = timeArray / SAMPLING_FREQ  # Second
    timeArray = timeArray * 1000  # Scale to milliseconds

    # Plot the sound signal in time domain
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.plot(timeArray, mono_signal, color='b')

    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()

def visualize_spectrogram(Sxx):
    # Convert to decibels
    Sxx_log = 10 * np.log10(Sxx)
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.show()


def read_zipped_mp3(file):
    # reads binary data and returns mono signal
    mp3_data = mp3_file.read()  # Read the binary data of the mp3 file
    audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3")
    mono_signal = audio.set_channels(1)
    s1 = mono_signal.get_array_of_samples()

    # Use max_value and normalize sound data to get values between -1 & +1
    max_value = np.max(np.abs(s1))
    s1 = s1/max_value

    return s1
# Genre mapping is needed for top-level genre labelling
genre_mapping_file = open(genre_mapping_directory)
genre_mapping = json.load(genre_mapping_file)

too_short = []
erronous = []
genre_not_present = []
count = 0
with h5py.File(hdf5_path, 'w') as h5f:
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        mp3_files = [file_name for file_name in zip_ref.namelist() if file_name.endswith('.mp3')]
        for idx, file_name in tqdm(enumerate(mp3_files), total=len(mp3_files), desc="Processing MP3 files"):
            try:
                with zip_ref.open(file_name) as mp3_file:
                    # get genre info
                    mp3 = MP3(mp3_file, ID3=ID3)
                    id3 = mp3.tags
                    child_genre = id3.get('TCON', 'Unknown Genre').text[0] if id3.get('TCON') else "NAN"

                    # This will disregard tracks with genres that are not recognized
                    genre = genre_mapping[child_genre]

                    mono_signal = read_zipped_mp3(mp3_file)
                    #visualize_audio(mono_signal)
                    if len(mono_signal) >= MIN_SAMPLE_LENGTH:
                        mono_signal = mono_signal[:MIN_SAMPLE_LENGTH-1]
                    else:
                        too_short.append(file_name)
                        continue

                    f, t, Sxx = signal.spectrogram(
                        mono_signal, 
                        fs=SAMPLING_FREQ, 
                        nperseg=512, 
                        noverlap=256
                    )
                    # Avoid zero for log scaling
                    Sxx = np.where(Sxx == 0, 1e-30, Sxx)
                    Sxx_db = 10 * np.log10(Sxx)
                    #Sxx_db = Sxx_db.astype(np.float)
                    #print(type(Sxx_db[0][0]),Sxx_db[0][:10] )

                    group = h5f.create_group(f'file_{idx}')
                    group.create_dataset('spectrogram', data=Sxx_db)
                    group.attrs['file_name'] = file_name
                    group.attrs['genre'] = genre

                    #visualize_spectrogram(Sxx)
                    if count == 1000:
                        break
                count += 1

            except Exception as e:
                erronous.append(file_name)
                print("While trying to process: ", file_name)
                print(e)
                print("-" * 40)

print(set(genre_not_present))