import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# DATA GENERATION
# -----------------------------
np.random.seed(42)
frames = 300
time = np.linspace(0, 30, frames)
external_temp = 35 + 2 * np.sin(0.1 * time)
occupancy = (np.sin(0.2 * time) > 0).astype(float)
temperature = 24 + 0.3 * np.sin(0.1 * time) + 0.2 * occupancy + np.random.normal(0, 0.2, frames)

# -----------------------------
# NORMALIZE FEATURES
# -----------------------------
features = np.column_stack([temperature, occupancy, external_temp])
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# SEQ2SEQ LSTM MODEL DEFINITION
# -----------------------------
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_len=10):
        super(Seq2SeqLSTM, self).__init__()
        self.output_len = output_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

# -----------------------------
# DATA PREPARATION
# -----------------------------
window = 30
output_len = 10
X, y = [], []
for i in range(frames - window - output_len):
    X.append(features_scaled[i:i+window])
    y.append([features_scaled[i+window+j, 0] for j in range(output_len)])

X = np.array(X)
y = np.array(y)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = Seq2SeqLSTM(output_len=output_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

# -----------------------------
# REAL-TIME ANIMATION SETUP
# -----------------------------
model.eval()
predictions = []
actuals = []
errors = []
time_steps = []

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, frames - window - output_len)
ax.set_ylim(22, 28)
actual_line, = ax.plot([], [], 'b-', label='Actual Temp')
pred_line, = ax.plot([], [], 'r--', label='Predicted Temp')
error_line, = ax.plot([], [], 'g-', label='Prediction Error')
ax.set_title("Sequence-to-Sequence AI Forecast (Digital Twin)")
ax.set_xlabel("Time Step")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(True)

def init():
    actual_line.set_data([], [])
    pred_line.set_data([], [])
    error_line.set_data([], [])
    return actual_line, pred_line, error_line

def update(frame):
    if frame >= len(X_tensor):
        return actual_line, pred_line, error_line

    with torch.no_grad():
        input_seq = X_tensor[frame:frame+1]
        pred_scaled = model(input_seq).squeeze().numpy()
        pred_unscaled = scaler.inverse_transform(np.column_stack([pred_scaled, np.zeros((output_len, 2))]))[:, 0]

    actual_unscaled = scaler.inverse_transform(
        np.column_stack([y[frame], np.zeros((output_len, 2))])
    )[:, 0]

    pred_mean = np.mean(pred_unscaled)
    actual_mean = np.mean(actual_unscaled)
    error = abs(pred_mean - actual_mean)

    time_steps.append(frame)
    predictions.append(pred_mean)
    actuals.append(actual_mean)
    errors.append(error)

    actual_line.set_data(time_steps, actuals)
    pred_line.set_data(time_steps, predictions)
    error_line.set_data(time_steps, errors)
    return actual_line, pred_line, error_line

ani = animation.FuncAnimation(fig, update, frames=len(X_tensor), init_func=init, interval=100, blit=False)
plt.tight_layout()
plt.show()
