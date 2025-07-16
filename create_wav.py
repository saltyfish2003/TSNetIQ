import numpy as np
from scipy.io import wavfile
# Set parameters
fs = 4200  # Sampling rate (Hz)
duration = 5  # Duration (seconds)
frequency = 2000  # Frequency (Hz)
# Generate time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
# Generate sine wave
signal = 0.5 * np.sin(2 * np.pi * frequency * t)
# Save as WAV file
wavfile.write("tone.wav", fs, signal.astype(np.float32))