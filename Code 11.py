import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Time simulation
frames = 100
time = np.linspace(0, 10, frames)

# Room state variables
temp = []
humidity = []

# Simulation parameters
external_temp = 35  # external temperature in Celsius
ac_power = -0.2     # degrees per second when AC is on
humidity_gain = 0.05
humidity_loss = -0.02
occupancy = np.zeros(frames)
occupancy[20:60] = 1  # people in the room during this interval

# Initialize
T0 = 24  # starting room temperature
H0 = 50  # starting humidity

# Figure setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.set_title("Digital Twin – Room Temperature")
ax1.set_xlim(0, 10)
ax1.set_ylim(15, 40)
temp_line, = ax1.plot([], [], 'r-', label='Temperature (°C)')
ax1.legend()

ax2.set_title("Room Humidity")
ax2.set_xlim(0, 10)
ax2.set_ylim(30, 80)
humid_line, = ax2.plot([], [], 'b-', label='Humidity (%)')
ax2.legend()

# Storage for plotting
temp_vals = []
humid_vals = []

def simulate_physical_room(t, T_prev, H_prev, occupied):
    # Simulate temperature
    T = T_prev + (external_temp - T_prev) * 0.01
    if occupied:
        T += 0.1  # people heat the room
    else:
        T += ac_power  # AC cools the room

    # Simulate humidity
    H = H_prev + (humidity_gain if occupied else humidity_loss)
    return T, H

def init():
    temp_line.set_data([], [])
    humid_line.set_data([], [])
    return temp_line, humid_line

def update(i):
    global T0, H0
    occ = occupancy[i]
    T0, H0 = simulate_physical_room(time[i], T0, H0, occ)
    temp_vals.append(T0)
    humid_vals.append(H0)
    temp_line.set_data(time[:i+1], temp_vals)
    humid_line.set_data(time[:i+1], humid_vals)
    return temp_line, humid_line

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, interval=200, blit=False)
plt.tight_layout()
plt.show()