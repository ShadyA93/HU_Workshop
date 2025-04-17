import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import IsolationForest
import pandas as pd

# Parameters
nx, ny = 20, 4
n_frames = 30
stress_warning_threshold = 1.2

# Storage for AI model
stress_series = []
labels = []

# Base mesh
mesh_base = np.ones((ny, nx))
warning_triggered = False

# Setup plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle("Digital Twin – AI-Based Beam Crack Monitoring and Prediction")

# Heatmap
im = ax1.imshow(np.zeros((ny, nx)), cmap='hot', interpolation='nearest', vmin=0, vmax=2)
ax1.set_title("Stress Distribution (Heatmap)")

# Line plot
line, = ax2.plot([], [], 'r-', lw=2)
ax2.set_xlim(0, nx)
ax2.set_ylim(0, 2.5)
ax2.set_title("Stress Along Middle Row")
text_box = ax2.text(0.02, 2.2, '', fontsize=10, color='blue')

# AI Prediction plot
ai_pred_plot, = ax3.plot([], [], 'g-', label="AI Detected Crack")
ai_true_plot, = ax3.plot([], [], 'k--', label="Actual Crack")
ax3.set_xlim(0, n_frames)
ax3.set_ylim(-0.1, 1.1)
ax3.set_title("AI-Based Crack Detection Over Time")
ax3.set_xlabel("Frame")
ax3.set_ylabel("Crack Detected")
ax3.legend()

# Data for AI plot
ai_pred_data = []
ai_true_data = []

def apply_load(mesh, frame):
    stress = np.random.rand(*mesh.shape) * mesh
    if frame >= 5:
        crack_x = min(frame - 5, nx - 1)
        crack_y = ny // 2
        mesh[crack_y, crack_x] = 0.1
        stress[crack_y, crack_x:] += 1.5
        return stress, 1
    return stress, 0

def init():
    im.set_array(np.zeros((ny, nx)))
    line.set_data([], [])
    ai_pred_plot.set_data([], [])
    ai_true_plot.set_data([], [])
    return im, line, ai_pred_plot, ai_true_plot, text_box

def update(frame):
    global warning_triggered

    mesh = mesh_base.copy()
    stress, label = apply_load(mesh, frame)
    mid_row = ny // 2
    mid_row_stress = stress[mid_row, :]

    # Store for training
    stress_series.append(mid_row_stress.tolist())
    labels.append(label)

    im.set_array(stress)
    line.set_data(np.arange(nx), mid_row_stress)

    # Train model after a few frames
    prediction = 0
    if frame >= 10:
        X = np.array(stress_series[:-1])
        model = IsolationForest(contamination=0.15).fit(X)
        prediction = int(model.predict([mid_row_stress])[0] == -1)

    # Update AI chart
    ai_pred_data.append(prediction)
    ai_true_data.append(label)

    ai_pred_plot.set_data(np.arange(len(ai_pred_data)), ai_pred_data)
    ai_true_plot.set_data(np.arange(len(ai_true_data)), ai_true_data)

    # Update status
    if prediction == 1 and not warning_triggered:
        text_box.set_text(f"🤖 AI Alert: Crack Detected at Frame {frame}")
        warning_triggered = True
    elif warning_triggered:
        text_box.set_text(f"Tracking Crack... Frame {frame}")
    else:
        text_box.set_text("Status: Healthy Beam")

    return im, line, ai_pred_plot, ai_true_plot, text_box

ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=600, blit=False)
plt.tight_layout()
plt.show()
