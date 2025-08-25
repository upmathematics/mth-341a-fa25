import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create two 2D vectors
v1 = np.array([2, 1])
v2 = np.array([1, 3])

# Step 2: Construct a symmetric matrix from outer products
A = np.outer(v1, v1) + np.outer(v2, v2)

# Step 3: Eigendecomposition
eigvals, eigvecs = np.linalg.eigh(A)
Q = eigvecs
Λ = np.diag(eigvals)

# Step 4: Unit circle
theta = np.linspace(0, 2*np.pi, 100)
unit_circle = np.array([np.cos(theta), np.sin(theta)]).T

# Step 5: Transformations
rotated = unit_circle @ Q
scaled = unit_circle @ Λ
rotated_scaled = unit_circle @ Q @ Λ @ Q.T

# Step 6: Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
titles = ['Original Vectors', 'Rotation Only (Q)', 'Scaling Only (Λ)', 'Rotation + Scaling (QΛQᵀ)']
circles = [unit_circle, rotated, scaled, rotated_scaled]

for ax, title, circle in zip(axs.flat, titles, circles):
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(title)

    # Plot unit circle or transformed version
    ax.plot(circle[:,0], circle[:,1], color='red', label='Transformed Circle')

    # Plot original vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green', label='v2')

    ax.legend()

plt.tight_layout()
plt.show()