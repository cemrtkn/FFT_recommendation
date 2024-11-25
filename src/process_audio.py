import zipfile
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TCON
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import h5py
import utils
from config import hdf5_path, dataset_path, tracks_metadata_path
import librosa


SAMPLING_FREQ = 44100
FIGURE_WIDTH = 7 
FIGURE_HEIGHT = 3.5  
MIN_SAMPLE_LENGTH = 1321967
# bit depth for 16 bit audio
BIT_DEPTH = 32768.0



def visualize_audio(mono_signal):
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

def plot_log_mel_spectrogram(log_mel, sampling_freq, hop_length, title="Log-Mel Spectrogram"):
    time_axis = np.arange(log_mel.shape[1]) * hop_length / sampling_freq
    mel_bins = np.arange(log_mel.shape[0])
    
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel, aspect='auto', origin='lower', interpolation='none', cmap='magma',
               extent=[time_axis[0], time_axis[-1], mel_bins[0], mel_bins[-1]])
    plt.colorbar(label='Log-Mel Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency Bins')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def read_zipped_mp3(file):
    # reads binary data and returns mono signal
    mp3_data = mp3_file.read()  # Read the binary data of the mp3 file
    audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3")
    mono_signal = audio.set_channels(1)
    s1 = mono_signal.get_array_of_samples()

    # Use max_value and normalize sound data to get values between -1 & +1
    s1 = np.divide(s1, BIT_DEPTH)

    return s1
def audio_to_spect(mono_signal,n_fft=2048 ,mel=True):
    spect_db = None
    hop_length = int(n_fft/4)

    stft = np.abs(librosa.stft(mono_signal, n_fft=2048, hop_length=hop_length))
    if mel:
        # taking it to the power of 2 makes all the difference!!!
        mel = librosa.feature.melspectrogram(sr=SAMPLING_FREQ, S=stft**2)
        spect_db = librosa.amplitude_to_db(mel)
        #plot_log_mel_spectrogram(spect_db, SAMPLING_FREQ, hop_length=512, title="Log-Mel Spectrogram")
    else:
        # Avoid zero for log scaling
        # based on min value in the first 2000 songs ~ 1e-54
        stft = np.where(stft == 0, 1e-55, stft)
        spect_db = 10 * np.log10(stft**2)
        #plot_log_mel_spectrogram(spect_db, SAMPLING_FREQ, hop_length=512, title="Log-Mel Spectrogram")
    return spect_db
    


tracks_metadata = utils.load(tracks_metadata_path)
small_metadata = tracks_metadata[tracks_metadata['set', 'subset'] <= 'small']
genre_labels = small_metadata[("track", "genre_top")]


too_short = []
erronous = []
genre_not_present = []
count = 0
with h5py.File(hdf5_path, 'w') as h5f:
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        mp3_files = [file_name for file_name in zip_ref.namelist() if file_name.endswith('.mp3')]
        for idx, file_name in tqdm(enumerate(mp3_files), total=len(mp3_files), desc="Processing MP3 files"):
            try:
                with zip_ref.open(file_name) as mp3_file:
                    genre = genre_labels.iloc[idx]

                    mono_signal = read_zipped_mp3(mp3_file)
                    if len(mono_signal) >= MIN_SAMPLE_LENGTH:
                        mono_signal = mono_signal[:MIN_SAMPLE_LENGTH-1]
                    else:
                        too_short.append(file_name)
                        continue

                    spect_db = audio_to_spect(mono_signal, mel=True)

                    group = h5f.create_group(f'file_{idx}')
                    group.create_dataset('spectrogram', data=spect_db)
                    group.attrs['file_name'] = file_name
                    group.attrs['genre'] = genre

                    if count == 2000:
                        break
                count += 1

            except Exception as e:
                erronous.append(file_name)
                print("While trying to process: ", file_name)
                print(e)
                print("-" * 40)

print(set(genre_not_present))