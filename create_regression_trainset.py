import os
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra
# Both trainset and testset can be originated by this file.
# Simulation parameters
N_MIC = 4  # Number of microphone array elements, change this to originate different M data
d0 = 0.02  # Element spacing (meters)
SNR = 5  # Signal-to-Noise Ratio, change this to originate different SNR data
fs, signal = wavfile.read("tone.wav")  # Load audio file
room_fs = fs # change this to originate different fs data
room_dim = [4, 4, 4]  # Room dimensions (meters), you can change by youself
# Calculate required radius for circular array to maintain element spacing
radius = d0 / (2 * np.sin(np.pi / N_MIC))
# Material properties near a anechoic chamber
materials = {
    "east": pra.Material(0.99),
    "west": pra.Material(0.99),
    "south": pra.Material(0.99),
    "north": pra.Material(0.99),
    "floor": pra.Material(0.99),
    "ceiling": pra.Material(0.99),
}
# Array center position, you can change by youself
mic_center = np.array([2, 2.5, 0.05])
# Construct circular array coordinates (x-y plane)
angles = np.linspace(0, 2 * np.pi, N_MIC, endpoint=False)
mic_offsets = np.stack([
    radius * np.cos(angles),
    radius * np.sin(angles),
    np.zeros(N_MIC)
], axis=1)
# Rotation function for array orientation
def rotate_points(points, angle_deg):
    angle_rad = np.radians(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return np.dot(points, R.T)
# Begin simulation loop
for i in range(10000):
    # Generate random array rotation angle
    angle = np.round(np.random.uniform(0, 360), 1)
    # Create room simulation environment
    room = pra.ShoeBox(room_dim, fs=room_fs, ray_tracing=True,
                       air_absorption=True, materials=materials)
    # Add sound source at fixed position
    room.add_source([2, 0, 1.5], signal=signal)
    # Rotate microphone array and position in room
    rotated_offsets = rotate_points(mic_offsets, angle)
    mic_locs = (mic_center + rotated_offsets).T
    room.add_microphone_array(mic_locs)
    # Run simulation
    room.simulate()
    mic_signals = room.mic_array.signals
    # Extract first 3 seconds of signals
    mic_signals_3s = mic_signals[:, :3 * fs]
    # Add white Gaussian noise based on SNR
    signal_power = np.mean(mic_signals_3s ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (SNR / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*mic_signals_3s.shape)
    mic_signals_noisy = mic_signals_3s + noise
    # Save simulated data/your path
    np.save(os.path.join("test3", f"{angle:.1f}_{i}.npy"), mic_signals_noisy)