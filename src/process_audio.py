import zipfile
import os
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm



SAMPLING_FREQ = 44100
FIGURE_WIDTH = 7 
FIGURE_HEIGHT = 3.5  

# Get the current working directory
current_directory = os.getcwd()
file_path = os.path.join(current_directory, '..', '..', 'data', 'fma_small.zip')
extract_directory = os.path.join(current_directory, '..', '..', 'data')

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
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
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


with zipfile.ZipFile(file_path, 'r') as zip_ref:
    count = 0
    min_length = 2000000
    mp3_files = [file_name for file_name in zip_ref.namelist() if file_name.endswith('.mp3')]

    for file_name in tqdm(mp3_files[4470:], desc="Processing MP3 files"):
        try:
            if file_name.endswith('.mp3'):
                with zip_ref.open(file_name) as mp3_file:
                    mono_signal = read_zipped_mp3(mp3_file)
                    #visualize_audio(mono_signal)

                    # Calculate the spectrogram using scipy.signal.spectrogram
                    '''f, t, Sxx = signal.spectrogram(
                        mono_signal, 
                        fs=SAMPLING_FREQ, 
                        nperseg=256, 
                        noverlap=128
                    )
                    print("Shape of Sxx (spectrogram data):", Sxx.shape)
                    visualize_spectrogram(Sxx)'''

                    
                    data_length = len(mono_signal)
                    if data_length < min_length:
                        min_length = data_length
                        print(min_length)
        except Exception:
            print(Exception)