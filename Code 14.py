import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd

# 1. Simulated 1D Heat Equation u_t = alpha * u_xx
alpha = 0.1
L = 1.0
T_max = 1.0
nx = 100
nt = 100

x = np.linspace(0, L, nx)
t = np.linspace(0, T_max, nt)
X, T = np.meshgrid(x, t)
u_true = np.exp(-np.pi**2 * alpha * T) * np.sin(np.pi * X)

# Prepare training data
X_train = torch.tensor(np.vstack([X.ravel(), T.ravel()]).T, dtype=torch.float32, requires_grad=True)
u_train = torch.tensor(u_true.ravel(), dtype=torch.float32).unsqueeze(1)

# 2. Shared NN Architecture
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

pinn_model = BaseNet()
nn_model = BaseNet()

# 3. PINN Physics Loss
def physics_loss(model, xyt, alpha=0.1):
    xyt.requires_grad_(True)
    u = model(xyt)
    grads = autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return torch.mean((u_t - alpha * u_xx)**2)

# 4. Train both models
epochs = 3000
pinn_optimizer = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

for epoch in range(epochs):
    # --- NN ---
    nn_optimizer.zero_grad()
    u_pred_nn = nn_model(X_train)
    loss_nn = torch.mean((u_pred_nn - u_train)**2)
    loss_nn.backward()
    nn_optimizer.step()

    # --- PINN ---
    pinn_optimizer.zero_grad()
    u_pred_pinn = pinn_model(X_train)
    loss_data = torch.mean((u_pred_pinn - u_train)**2)
    loss_phys = physics_loss(pinn_model, X_train, alpha)
    loss_pinn = 0.2 * loss_data + 1.0 * loss_phys
    loss_pinn.backward()
    pinn_optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch} | NN Loss: {loss_nn.item():.4e} | PINN Loss: {loss_pinn.item():.4e} (Data: {loss_data.item():.4e}, Phys: {loss_phys.item():.4e})")

# 5. Predict on full domain
X_eval = torch.tensor(np.vstack([X.ravel(), T.ravel()]).T, dtype=torch.float32)
with torch.no_grad():
    u_pred_nn = nn_model(X_eval).numpy().reshape((nt, nx))
    u_pred_pinn = pinn_model(X_eval).numpy().reshape((nt, nx))

mse_nn = np.mean((u_pred_nn - u_true)**2)
mse_pinn = np.mean((u_pred_pinn - u_true)**2)

# 6. Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(u_true, extent=[0, L, 0, T_max], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("True u(x, t)")

plt.subplot(1, 3, 2)
plt.imshow(u_pred_nn, extent=[0, L, 0, T_max], origin='lower', aspect='auto', cmap='plasma')
plt.colorbar()
plt.title(f"NN Prediction\nMSE: {mse_nn:.2e}")

plt.subplot(1, 3, 3)
plt.imshow(u_pred_pinn, extent=[0, L, 0, T_max], origin='lower', aspect='auto', cmap='inferno')
plt.colorbar()
plt.title(f"PINN Prediction\nMSE: {mse_pinn:.2e}")

plt.tight_layout()
plt.show()
