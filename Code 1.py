import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate time and data
time = np.linspace(0, 20, 500)  # Reduced frames for smaller file size
sensor_data = np.sin(2 * np.pi * 1 * time) + 0.2 * np.random.randn(len(time))
twin_prediction = np.sin(2 * np.pi * 1 * time)

# Setup the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 20)
ax.set_ylim(-2, 2)
ax.set_title("Real-Time Digital Twin Monitoring")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# Create empty line objects
line1, = ax.plot([], [], label="Sensor Data (Noisy)", color='tab:blue')
line2, = ax.plot([], [], label="Digital Twin Prediction", color='tab:orange')
ax.legend()

# Initialize function for the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Animation update function
def animate(i):
    x = time[:i]
    y1 = sensor_data[:i]
    y2 = twin_prediction[:i]
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    ax.set_title(f"Real-Time Digital Twin Monitoring - Frame {i}")
    return line1, line2

# Create animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(time), interval=40, blit=True
)

# Save animation as GIF using Pillow
ani.save("digital_twin_monitoring.gif", writer="pillow", fps=25)

# Display the plot window 
plt.show()