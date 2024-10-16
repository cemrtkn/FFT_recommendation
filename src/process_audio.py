import zipfile
import os
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

figure_width = 7  # inches
figure_height = 3.5  # inches

# Get the current working directory
current_directory = os.getcwd()
file_path = os.path.join(current_directory, '..', '..', 'data', 'fma_small.zip')
extract_directory = os.path.join(current_directory, '..', '..', 'data')


# Open the zip file
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith('.mp3'):
            with zip_ref.open(file_name) as mp3_file:
                mp3_data = mp3_file.read()  # Read the binary data of the mp3 file
                
                # Use BytesIO to treat the binary data as a file-like object
                audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3")
                
                # Get sampling frequency and raw audio data
                sampling_freq = audio.frame_rate
                sound_data = audio.get_array_of_samples()

                # Print or process the audio data
                print(f'Processed {file_name}:')
                print(f'Sampling Frequency: {sampling_freq}')
                print(f'Sound Data Length: {len(sound_data)}')
                print('-' * 40)

                # Determine the maximum values
                max_value = np.max(np.abs(sound_data))
                print(f'Max Value is {max_value}')
                # Use max_value and normalize sound data to get values between -1 & +1
                sound_data = sound_data/max_value
                # Lets just take a single audio channel (mono) even if it is stereo
                if len(sound_data.shape) == 1:
                    s1 = sound_data
                else:
                    s1 = sound_data[:, 0]

                # Get time domain representation of the sound pressure waves
                timeArray = np.arange(0, s1.shape[0], 1.0)
                timeArray = timeArray / sampling_freq  # Second
                timeArray = timeArray * 1000  # Scale to milliseconds

                # Plot the sound signal in time domain
                plt.figure(figsize=(figure_width, figure_height))
                plt.plot(timeArray, sound_data, color='b')

                plt.xlabel('Time [ms]')
                plt.ylabel('Amplitude', fontsize=12)
                plt.grid()
                plt.tight_layout()
                plt.show()

                break
