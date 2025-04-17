import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Beam mesh
nx, ny = 20, 4
mesh_base = np.ones((ny, nx))

# Crack propagation log and detection state
warning_triggered = False
crack_log = []

# Threshold for abnormal stress
stress_warning_threshold = 1.2

def apply_load(mesh, frame):
    # Simulate a healthy or cracked mesh
    stress = np.random.rand(*mesh.shape) * mesh
    if frame >= 5:  # Crack starts at frame 5
        crack_x = min(frame - 5, nx - 1)
        crack_y = ny // 2
        mesh[crack_y, crack_x] = 0.1  # Simulate crack
        stress[crack_y, crack_x:] += 1.5  # Localised stress spike
        crack_log.append((frame, crack_x))
    return stress

# Setup plot with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle("Digital Twin – Beam Crack Detection and Propagation Monitoring")

# Heatmap (ax1)
im = ax1.imshow(np.zeros_like(mesh_base), cmap='hot', interpolation='nearest', vmin=0, vmax=2)
cb = plt.colorbar(im, ax=ax1)
ax1.set_title("Stress Distribution Heatmap")
ax1.set_ylabel("Height")
ax1.set_xlabel("Length")

# Stress Line Plot (ax2)
line, = ax2.plot([], [], 'r-', lw=2)
ax2.set_xlim(0, nx)
ax2.set_ylim(0, 2.5)
ax2.set_title("Stress Along Middle Row (Monitoring Zone)")
ax2.set_ylabel("Stress")
ax2.set_xlabel("Beam Length")

# Annotation for warning
text_box = ax2.text(0.02, 2.2, '', fontsize=12, color='red')

def update(frame):
    global warning_triggered
    mesh = mesh_base.copy()
    stress = apply_load(mesh, frame)
    
    im.set_array(stress)

    mid_row = ny // 2
    line.set_data(np.arange(nx), stress[mid_row, :])

    # Check for abnormal stress
    if not warning_triggered and np.any(stress[mid_row, :] > stress_warning_threshold):
        text_box.set_text(f"⚠️ Warning: Crack Detected at Frame {frame}")
        warning_triggered = True
    elif warning_triggered:
        text_box.set_text(f"Tracking Crack... Frame {frame}")
    else:
        text_box.set_text("Status: Healthy Beam")

    return im, line, text_box

# Animate
ani = animation.FuncAnimation(fig, update, frames=nx + 5, interval=500, blit=False)
plt.tight_layout()
plt.show()
