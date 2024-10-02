import pyaudio   # xtype: ignore # Soundcard audio I/O access library
import wave  # Pythocyn 3 module for reading/writing simple .wav files
import struct
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.signal  # type: ignore
from collections import Counter

# Setup channel info
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Adjust to your number of channels
RATE = 44100  # Sample Rate 44.1kHz
CHUNK = 1024  # Block Size
RECORD_SECONDS = 9  # Record time
WAVE_OUTPUT_FILENAME = "file.wav"
LOWEST_FREQ = 7
TRANSFORM_FREQ = 10

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

#lowest_peak_freq_array = [] §
#volume_100ml = []
#coeffecients = np.polyfit(lowest_peak_freq_array,volume_100ml,3)
#polynom = np.poly_1d(coeffecients)

plt.imshow(time_series_np, cmap="viridis", interpolation="bilinear", origin="lower",
           vmax=np.average(abs(time_series_np)) * 10, vmin=0)
freq = np.fft.fftfreq(width_in_spl, d=1/RATE)[:freq_width]
plt.yticks(ticks=range(freq_width)[::50], labels=freq[::50])
plt.xticks(ticks=range(0, RECORD_SECONDS * 100)[::50], labels=range(0, RECORD_SECONDS * 1000, 10)[::50])
plt.xlabel("Time in ms")
plt.ylabel("Frequency in Hz")
plt.show()
