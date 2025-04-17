import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import IsolationForest

# Step 1: Generate Historical Data (normal)
x_hist = np.linspace(0, 10, 500).reshape(-1, 1)
y_hist = np.sin(x_hist) + 0.05 * np.random.randn(*x_hist.shape)
train_data = np.hstack([x_hist, y_hist])

# Step 2: Train Isolation Forest on normal signal patterns
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(train_data)

# Step 3: Generate Live Data (with anomalies)
x_live = np.linspace(0, 10, 500).reshape(-1, 1)
y_live = np.sin(x_live) + 0.05 * np.random.randn(*x_live.shape)
y_live[300:320] += 1.5  # Inject anomaly
y_live[400:410] -= 1.5  # Inject anomaly
live_data = np.hstack([x_live, y_live])

# Step 4: Setup the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-3, 3)
line_normal, = ax.plot([], [], 'bo', label="Normal", markersize=2)
line_anomaly, = ax.plot([], [], 'ro', label="Anomaly", markersize=4)
ax.set_title("Live Anomaly Detection with Isolation Forest")
ax.set_xlabel("x")
ax.set_ylabel("Signal")
ax.legend(loc='upper right')

# Step 5: Update function for animation
def update(frame):
    if frame == 0:
        return line_normal, line_anomaly

    current_x = x_live[:frame]
    current_y = y_live[:frame]
    current_features = np.hstack([current_x, current_y])

    preds = model.predict(current_features)
    normal_idx = preds == 1
    anomaly_idx = preds == -1

    line_normal.set_data(current_x[normal_idx], current_y[normal_idx])
    line_anomaly.set_data(current_x[anomaly_idx], current_y[anomaly_idx])
    ax.set_title(f"Live Anomaly Detection - Frame {frame}")
    return line_normal, line_anomaly

# Step 6: Animate and Save
ani = animation.FuncAnimation(
    fig, update, frames=range(1, len(x_live)), interval=20, blit=True
)

# Save as GIF
ani.save("live_anomaly_detection.gif", writer="pillow", fps=25)

# Optionally show the last frame
plt.show()
