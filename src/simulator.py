
'''
Warehouse simulator
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Coordinates for zones (Z) and aisles (C) as NumPy arrays
W, H = 500, 500
Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
C = [[150, 250], [250, 250], [350, 250]]
C_last_dir = [[0, 0], [0, 0], [0, 0]]
V = [6] * 3
O = [[0, 1], [3, 5], [4, 2]]

# To numpy
Z = np.array(Z, dtype=np.dtype('float32'))
C = np.array(C, dtype=np.dtype('float32'))
V = np.array(V, dtype=np.dtype('float32'))
C_last_dir = np.array(C_last_dir, dtype=np.dtype('float32'))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal', adjustable='box')  # Keep equal scaling for x and y axes

# Update function for animation
def update(frame):
    ax.clear()
    ax.set_title("Warehouse Simulator")

    # Zones
    for idx, z in enumerate(Z):
        ax.scatter(z[0], z[1], color='gray', marker='s', s=400)
        ax.text(z[0], z[1], f'Z{idx}', color='white', fontsize=10, ha='center', va='center')

    # AGVs
    for idx, c in enumerate(C):
        if len(O[idx]):
            dest = Z[O[idx][0]]
            direction = dest - C[idx]
            direction_normalized = (direction / np.linalg.norm(direction)) if direction.any() else np.array([0, 0])
            C[idx] += direction_normalized * V[idx]
            if np.isin(-1, (np.sign(C_last_dir[idx]) * np.sign(direction_normalized))) or (C[idx] == dest).all():
                C[idx] = dest
                O[idx].pop(0)
            C_last_dir[idx] = direction_normalized

            # Plot arrow
            if (C[idx] != dest).any():
                ax.arrow(
                    c[0], c[1], direction_normalized[0] * 5 * V[idx],
                    direction_normalized[1] * 5 * V[idx],
                    head_width=5,
                    head_length=7,
                    fc=('green' if len(O[idx]) == 0 else 'red'),
                    ec=('green' if len(O[idx]) == 0 else 'red')
                )

        # Plot AGV with label
        ax.scatter(c[0], c[1], color=('green' if len(O[idx]) == 0 else 'red'), marker='o', s=200)
        ax.text(c[0], c[1], f'C{idx}', color='white', fontsize=8, ha='center', va='center')
    
    # Axes style
    ax.set_xticks(range(0, W + 1, W // 10))
    ax.set_yticks(range(0, H + 1, H // 10))
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
plt.show()
