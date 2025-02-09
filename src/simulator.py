
'''
Warehouse simulator
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Warehouse:
    '''
    Warehouse simulation
    '''
    def __init__(self, W: float, H: float, Z: list[list[float]], C: list[list[float]], V: list[float]):
        self._W = W
        self._H = H
        self._Z = np.array(Z, dtype=np.dtype('float32'))
        self._C = np.array(C, dtype=np.dtype('float32'))
        self._V = np.array(V, dtype=np.dtype('float32'))
        self._C_last_dir = np.array([[0, 0] for _ in self._C], dtype=np.dtype('float32'))
        self._O = [[] for _ in self._C]

        # Create the figure and axis
        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        self._ax.set_xlim(0, W)
        self._ax.set_ylim(0, H)
        self._ax.set_aspect('equal', adjustable='box')  # Keep equal scaling for x and y axes

    def start(self):
        '''
        Start simulation
        '''
        self._anim = animation.FuncAnimation(self._fig, self._update, frames=100, interval=50)
        plt.show()

    def assign_job(self, c: int, o: list[int]):
        '''
        Assign a job to a specified AGV
        '''
        self._O[c] += o

    # Update function for animation
    def _update(self, frame):
        self._ax.clear()
        self._ax.set_title("Warehouse Simulator")

        # Zones (draw first so AGVs appear on top)
        for idx, z in enumerate(self._Z):
            self._ax.scatter(z[0], z[1], color='gray', marker='s', s=400, zorder=1)
            self._ax.text(z[0], z[1], f'Z{idx}', color='white', fontsize=10, ha='center', va='center', zorder=2)

        # AGVs (draw after zones so they appear on top)
        for idx, c in enumerate(self._C):
            if len(self._O[idx]):
                dest = self._Z[self._O[idx][0]]
                direction = dest - self._C[idx]
                direction_normalized = (direction / np.linalg.norm(direction)) if direction.any() else np.array([0, 0])
                self._C[idx] += direction_normalized * self._V[idx]
                if np.isin(-1, (np.sign(self._C_last_dir[idx]) * np.sign(direction_normalized))) or (self._C[idx] == dest).all():
                    self._C[idx] = dest
                    direction_normalized = np.array([0, 0])
                    self._O[idx].pop(0)
                self._C_last_dir[idx] = direction_normalized

                # Plot arrow
                if (self._C[idx] != dest).any():
                    self._ax.arrow(
                        c[0], c[1], direction_normalized[0] * 5 * self._V[idx],
                        direction_normalized[1] * 5 * self._V[idx],
                        head_width=5,
                        head_length=7,
                        fc=('green' if len(self._O[idx]) == 0 else 'red'),
                        ec=('green' if len(self._O[idx]) == 0 else 'red'),
                        zorder=3
                    )

            # Plot AGV with label
            self._ax.scatter(c[0], c[1], color=('green' if len(self._O[idx]) == 0 else 'red'), marker='o', s=200, zorder=4)
            self._ax.text(c[0], c[1], f'C{idx}', color='white', fontsize=8, ha='center', va='center', zorder=5)
        
        # Axes style
        self._ax.set_xticks(range(0, self._W + 1, self._W // 10))
        self._ax.set_yticks(range(0, self._H + 1, self._H // 10))
        self._ax.xaxis.set_tick_params(width=0)
        self._ax.yaxis.set_tick_params(width=0)
        self._ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

import time
import threading

W, H = 500, 500
Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
C = [[150, 250], [250, 250], [350, 250]]
V = [6] * 3
w = Warehouse(W, H, Z, C, V)

def warehouse_simulation():
    w.start()

def AGV_manager():
    O = [[0, 1], [3, 5], [4, 2], [2, 5], [0, 5]]
    time.sleep(2)
    w.assign_job(0, O[0])
    time.sleep(4)
    w.assign_job(1, O[1])
    time.sleep(2)
    w.assign_job(2, O[2])
    time.sleep(1)
    w.assign_job(1, O[3])
    time.sleep(0)
    w.assign_job(0, O[4])

def main():
    sim_thread = threading.Thread(target=AGV_manager, daemon=True)
    sim_thread.start()
    warehouse_simulation()
    

if __name__ == '__main__': # Entry point
    main()
