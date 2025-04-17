import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis

# -------------------------------
# SYSTEM PARAMETERS
# -------------------------------
fs = 1000
T = 1.0
N = int(T * fs)
time = np.linspace(0, T, N)
base_freq = 50
fault_freq = 200
frames_total = 100
fault_start_frame = 50

# -------------------------------
# FUNCTIONS
# -------------------------------

def generate_signal(frame):
    """Simulate shaft vibration signal with or without fault."""
    base_signal = np.sin(2 * np.pi * base_freq * time)
    noise = 0.3 * np.random.randn(N)
    if frame >= fault_start_frame:
        fault = 0.7 * np.sin(2 * np.pi * fault_freq * time)
        return base_signal + noise + fault, 1
    return base_signal + noise, 0

def extract_features(signal):
    """Extract robust FFT-based features."""
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(N, d=1/fs)
    feature_vector = [
        np.mean(fft_vals),
        np.std(fft_vals),
        np.max(fft_vals),
        skew(fft_vals),
        kurtosis(fft_vals)
    ]
    return np.array(feature_vector)

def moving_average(arr, window_size=3):
    """Smooth predictions using a moving average."""
    if len(arr) < window_size:
        return arr
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

# -------------------------------
# PLOT SETUP
# -------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("AI-Enhanced Digital Twin – Shaft Vibration Monitoring", fontsize=14)

# Time-domain
ax1.set_title("Time-Domain Vibration Signal")
ax1.set_xlim(0, T)
ax1.set_ylim(-2, 2)
line1, = ax1.plot([], [], lw=1.5)

# FFT Spectrum
ax2.set_title("Frequency Spectrum (FFT)")
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 50)
line2, = ax2.plot([], [], lw=1.5, color='orange')

# Fault Detection Plot
ax3.set_title("Smoothed AI Fault Detection vs Actual")
ax3.set_xlim(0, frames_total)
ax3.set_ylim(-0.1, 1.1)
ai_line, = ax3.plot([], [], 'r-', label="AI Prediction")
actual_line, = ax3.plot([], [], 'g--', label="Actual Fault")
ax3.set_xlabel("Frame")
ax3.set_ylabel("Fault Detected")
ax3.legend()

# -------------------------------
# DATA STORAGE
# -------------------------------
signal_log = []
features_log = []
actual_faults = []
ai_predictions = []
smoothed_preds = []
model = IsolationForest(contamination=0.1)
model_trained = False

# -------------------------------
# INIT FUNCTION
# -------------------------------
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    ai_line.set_data([], [])
    actual_line.set_data([], [])
    return line1, line2, ai_line, actual_line

# -------------------------------
# UPDATE FUNCTION
# -------------------------------
def update(frame):
    global model_trained

    # Simulate signal and ground truth
    signal, label = generate_signal(frame)
    signal_log.append(signal)
    actual_faults.append(label)

    # Time-domain plot
    line1.set_data(time, signal)

    # FFT Spectrum
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(N, d=1/fs)
    line2.set_data(fft_freq, fft_vals)

    # Feature extraction
    features = extract_features(signal)
    features_log.append(features)

    # Train AI model on healthy data
    if frame == 30:
        X_train = np.array(features_log[:30])
        model.fit(X_train)
        model_trained = True

    # Predict faults with Isolation Forest
    if model_trained:
        pred = int(model.predict(features.reshape(1, -1))[0] == -1)
    else:
        pred = 0
    ai_predictions.append(pred)

    # Apply moving average smoothing
    smoothed = moving_average(ai_predictions, window_size=5)
    if len(smoothed) < len(ai_predictions):
        smoothed_preds.append(smoothed[-1])
    else:
        smoothed_preds.append(pred)

    # Update plots
    ai_line.set_data(np.arange(len(smoothed_preds)), smoothed_preds)
    actual_line.set_data(np.arange(len(actual_faults)), actual_faults)

    return line1, line2, ai_line, actual_line

# -------------------------------
# RUN ANIMATION
# -------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=frames_total, init_func=init, interval=400, blit=False
)
plt.tight_layout()
plt.show()
