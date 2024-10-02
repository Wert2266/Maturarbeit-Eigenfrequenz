import pyaudio  # Soundcard audio I/O access library
import wave  # Python 3 module for reading/writing simple .wav files
import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from collections import Counter
# Setup channel info
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Adjust to your number of channels
RATE = 44100  # Sample Rate 44.1kHz
CHUNK = 1024  # Block Size
RECORD_SECONDS = 8  # Record time
WAVE_OUTPUT_FILENAME = "file.wav"
LOWEST_FREQ = 7
TRANSFORM_FREQ = 10
lowest_peak_freq_array = [1085,1043,1015,938,826,707,602]
volume_100ml = [0,100,200,300,400,500,600]
# Startup pyaudio instance
audio = pyaudio.PyAudio()
# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording...")
frames = []
all_data = []

# Record for RECORD_SECONDS
for i in range(0, int((RATE / CHUNK) * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    unp = list(map(lambda val: val[0], struct.iter_unpack('h', data)))
    all_data.extend(unp)
    frames.append(data)
print("Finished recording")

# Stop recording
stream.stop_stream()
stream.close()
audio.terminate()


waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

npdata = np.array(all_data, dtype=np.short)
width_in_s = 1 / LOWEST_FREQ
width_in_spl = int(RATE * width_in_s)
step_in_s = 1 / TRANSFORM_FREQ / 10
step_in_spl = int(RATE * step_in_s)

print('Sizes:', step_in_spl, width_in_spl)

offset = 0
freq_width = 1000
time_series = []

while offset + width_in_spl < npdata.shape[0]:
    slice_data = npdata[offset:offset + width_in_spl]
    sp = np.fft.fft(slice_data)
    time_series.append(np.abs(sp[:freq_width]))
    offset += step_in_spl

time_series_np = np.array(time_series).T
peaks_array = [scipy.signal.find_peaks(i)[0] for i in time_series_np]

# Find the peak frequency
peak_indices = np.argmax(time_series_np, axis=0)
peak_frequencies = [np.fft.fftfreq(width_in_spl, d=1/RATE)[idx] for idx in peak_indices[:freq_width]]

# Print peak frequencies
dict_peaks = Counter(peak_frequencies)
print("dict_peaks", dict_peaks)
threshold = 180  
lowest_peak_freq = None
for freq in sorted(dict_peaks.keys()):
    if dict_peaks[freq] > threshold:  # Threshold to filter out noise peaks
        lowest_peak_freq = freq
        break
print("Lowest modal frequency: ", lowest_peak_freq)

coeffecients = np.polyfit(lowest_peak_freq_array,volume_100ml,6)
polynom = np.poly1d(coeffecients)
print(str(polynom(lowest_peak_freq))+"ml")
