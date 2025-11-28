import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from src.analysis import *


# Create time array in milliseconds (matching your data format)
sampling_rate = 1000  # Hz
duration = 1.0  # seconds
time_array = np.arange(0, duration * 1000, 1000 / sampling_rate)  # in ms

# Create 15 channels with sinusoids at 6, 12, 18, ..., 90 Hz
n_channels = 15
bipolar_signals = {}
channel_labels = []
channel_depths = []

for i in range(n_channels):
    freq = (i + 1) * 6  # 6, 12, 18, 24, ..., 90 Hz
    
    # Create sinusoid at the specific frequency
    # time_array is in ms, so convert to seconds for frequency calculation
    signal = np.sin(2 * np.pi * freq * (time_array / 1000))
    
    # Add some noise for realism
    signal += np.random.normal(0, 0.1, len(signal))
    
    # Store in dictionary (matching your format)
    bipolar_signals[i] = signal
    
    # Create labels
    channel_labels.append(f"{freq} Hz")
    
    # Create fake depths (evenly spaced)
    channel_depths.append(i * 100)  # 0, 100, 200, ... μm

# Now test your function
fig_wvlt = plot_wavelet_transform(bipolar_signals, channel_labels, channel_depths, 
                                   time_array, time_range=(0, 1000))
plt.show()

print(f"Created {n_channels} channels with frequencies:")
for i in range(n_channels):
    print(f"  Channel {i}: {(i+1)*6} Hz")