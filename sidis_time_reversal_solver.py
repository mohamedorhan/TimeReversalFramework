# ============================================================================
# Sidis Time-Reversal Framework
# A Rigorous, Closed-Form, Multi-Level Simulation of Time Reversal Physics
# Author: Mohamed Orhan Zeinel | Inspired by William James Sidis
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.constants import hbar
from scipy.spatial.distance import euclidean

# ============================================================================
# 1. Classical Particle Simulation: Forward and Reversed Time
# ============================================================================
N = 20                 # Number of particles
T = 100                # Total simulation time
dt = 0.1               # Time step
steps = int(T / dt)

np.random.seed(42)     # Reproducibility

# Initial positions and velocities
pos_fwd = np.random.rand(N, 2) * 10
vel_fwd = np.random.randn(N, 2)
pos_rev = pos_fwd.copy()
vel_rev = -vel_fwd.copy()

traj_fwd = np.zeros((steps, N, 2))
traj_rev = np.zeros((steps, N, 2))

# Forward time evolution
for t in range(steps):
    pos_fwd += vel_fwd * dt
    traj_fwd[t] = pos_fwd

# Reversed time evolution
for t in range(steps):
    pos_rev += vel_rev * dt
    traj_rev[t] = pos_rev

# ============================================================================
# 2. Entropy Calculation Over Time
# ============================================================================
def compute_entropy(traj, grid_size=10):
    entropies = []
    for t in range(traj.shape[0]):
        grid = np.zeros((grid_size, grid_size))
        indices = np.clip((traj[t] / 10 * grid_size).astype(int), 0, grid_size - 1)
        for x, y in indices:
            grid[x, y] += 1
        probs = grid.flatten()
        probs = probs[probs > 0] / np.sum(probs)
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)
    return np.array(entropies)

entropy_fwd = compute_entropy(traj_fwd)
entropy_rev = compute_entropy(traj_rev)

# ============================================================================
# 3. Chaos Analysis via Pairwise Distance Approximation (Lyapunov-like)
# ============================================================================
def avg_pairwise_dist(traj):
    steps, N, _ = traj.shape
    dists = []
    for t in range(steps):
        total = 0
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                total += euclidean(traj[t, i], traj[t, j])
                count += 1
        dists.append(total / count)
    return np.log(np.array(dists) + 1e-8)  # Log-scale for chaos approximation

lyap_fwd = avg_pairwise_dist(traj_fwd)
lyap_rev = avg_pairwise_dist(traj_rev)

# ============================================================================
# 4. Quantum Wave Packet Reversal (Time-Symmetric Schrödinger Evolution)
# ============================================================================
x_vals = np.linspace(-10, 10, 1024)
dx = x_vals[1] - x_vals[0]
k_vals = fftfreq(len(x_vals), d=dx) * 2 * np.pi

# Initial Gaussian wave packet
x0 = -3.0
k0 = 5.0
sigma = 1.0
psi0 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x_vals - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x_vals)

def evolve(psi, k, t, m=1.0):
    psi_k = fft(psi)
    E_k = (hbar**2 * k**2) / (2 * m)
    psi_k_t = psi_k * np.exp(-1j * E_k * t / hbar)
    return ifft(psi_k_t)

# Time evolution at +t and -t
psi_t = evolve(psi0, k_vals, 1.0)
psi_neg_t = evolve(psi0, k_vals, -1.0)

# ==============================
# 5. Final Visualization: Entropy, Chaos, Quantum Symmetry
# ==============================
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# (A) Entropy Evolution
axs[0].plot(np.arange(len(entropy_fwd)) * dt, entropy_fwd, label="Entropy (Forward)")
axs[0].plot(np.arange(len(entropy_rev)) * dt, entropy_rev, '--', label="Entropy (Reverse)")
axs[0].set_title("Entropy Evolution Over Time")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Entropy S")
axs[0].legend()
axs[0].grid(True)

# (B) Chaos Analysis
axs[1].plot(np.arange(len(lyap_fwd)) * dt, lyap_fwd, label="Chaos (Forward)")
axs[1].plot(np.arange(len(lyap_rev)) * dt, lyap_rev, '--', label="Chaos (Reverse)")
axs[1].set_title("Pairwise Chaos Indicator (Log Distance)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("log(Distance)")
axs[1].legend()
axs[1].grid(True)

# (C) Quantum Symmetry
axs[2].plot(x_vals, np.abs(psi_t)**2, label=r'$|\psi(x, +t)|^2$')
axs[2].plot(x_vals, np.abs(psi_neg_t)**2, '--', label=r'$|\psi(x, -t)|^2$')
axs[2].set_title("Quantum Time-Reversal of Gaussian Wave Packet")
axs[2].set_xlabel("Position x")
axs[2].set_ylabel("Probability Density")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("sidis_simulation_results.png", dpi=300)  
plt.show()
