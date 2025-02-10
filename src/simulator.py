
import pygame
import time
import threading
import numpy as np
import math
from orders_monitor import OrdersMonitor

class Warehouse:
    '''
    Warehouse Simulator
    '''
    _GRID_COLOR = (50, 50, 50)
    _WIDTH, _HEIGHT = 500, 500
    _BG_COLOR = (30, 30, 30)
    _ZONE_COLOR = (100, 100, 100)
    _TEXT_COLOR = (30, 30, 30)
    _AGV_COLOR_IDLE = (0, 255, 0)
    _AGV_COLOR_BUSY = (255, 0, 0)

    def __init__(self, W, H, Z, C, V):
        self._init_graphics()
        self._W = W
        self._H = H
        self._Z = np.array(Z, dtype=int)  # Zone coordinates
        self._C = np.array(C, dtype=float)  # AGV coordinates
        self._V = np.array(V, dtype=float)  # AGV speeds
        self._O_monitor = OrdersMonitor()
        self._O = [[] for _ in self._C]  # Orders for each AGV
        self._running = True

    def start(self):
        '''
        Starts the simulation
        '''
        clock = pygame.time.Clock()
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
            self._O_monitor.agv_start()
            self._update()
            self._draw()
            self._O_monitor.agv_end()
            clock.tick(30)  # 30 FPS

    def assign_job(self, c, o):
        '''
        Assigns a job to an AGV with specified values
        '''
        self._O_monitor.manager_start()
        self._O[c] += o
        self._O_monitor.manager_end()
    
    def _init_graphics(self):
        pygame.init()
        self._screen = pygame.display.set_mode((self._WIDTH, self._HEIGHT))  # Define the window here
        pygame.display.set_caption("Warehouse Simulator")
        self._font = pygame.font.Font(None, 24)
    
    def _update(self):

        for idx, c in enumerate(self._C):
            # start
            if self._O[idx]:
                dest = self._Z[self._O[idx][0]]
                direction = dest - c
                distance = np.linalg.norm(direction)
                
                if distance > self._V[idx]:
                    shift = (direction / distance) * self._V[idx]
                    self._C[idx] += shift
                else:
                    self._C[idx] = dest
                    self._O[idx].pop(0)
            # end
        
    
    def _draw_grid(self):
        # Draw the grid
        grid_size = 50
        for x in range(0, self._W, grid_size):
            pygame.draw.line(self._screen, self._GRID_COLOR, (x, 0), (x, self._H))
        for y in range(0, self._H, grid_size):
            pygame.draw.line(self._screen, self._GRID_COLOR, (0, y), (self._W, y))
    
    def _draw(self):
        self._screen.fill(self._BG_COLOR)
        
        # Draw the grid
        self._draw_grid()
        
        # Draw zones with labels
        for idx, z in enumerate(self._Z):
            pygame.draw.rect(self._screen, self._ZONE_COLOR, (int(z[0]) - 10, int(z[1]) - 10, 20, 20))
            
            # Create the label for the zone
            label = self._font.render(f"Z{idx}", True, self._TEXT_COLOR)

            # Calculate the position of the text centered above the zone
            label_width = label.get_width()
            label_height = label.get_height()
            x_pos = int(z[0] - label_width / 2)
            y_pos = int(z[1] - label_height / 2) - 15  # Move slightly above the zone

            self._screen.blit(label, (x_pos, y_pos))
        
        # Draw AGVs and direction arrows
        for idx, c in enumerate(self._C):
            color = self._AGV_COLOR_BUSY if self._O[idx] else self._AGV_COLOR_IDLE

            # Draw the arrow before the AGVs
            if self._O[idx]:
                destination = self._Z[self._O[idx][0]]
                angle = math.atan2(destination[1] - c[1], destination[0] - c[0])

                # Create the arrow
                arrow_length = 20
                arrow_width = 5
                arrow_points = [
                    (c[0] + math.cos(angle) * arrow_length, c[1] + math.sin(angle) * arrow_length),
                    (c[0] + math.cos(angle + math.pi / 6) * arrow_width, c[1] + math.sin(angle + math.pi / 6) * arrow_width),
                    (c[0] + math.cos(angle - math.pi / 6) * arrow_width, c[1] + math.sin(angle - math.pi / 6) * arrow_width),
                ]
                pygame.draw.polygon(self._screen, color, arrow_points)  # Use the same color as the AGV

            # Then draw the AGV over the arrow
            pygame.draw.circle(self._screen, color, (int(c[0]), int(c[1])), 10)
            
            # Calculate the AGV number to display in the center
            label = self._font.render(f"{idx}", True, self._TEXT_COLOR)

            # Calculate the position of the text centered inside the AGV
            label_width = label.get_width()
            label_height = label.get_height()
            x_pos = int(c[0] - label_width / 2)
            y_pos = int(c[1] - label_height / 2)

            self._screen.blit(label, (x_pos, y_pos))

        pygame.display.flip()

# AGV manager assigning jobs at different times
def AGV_manager(warehouse):
    O = [[0, 1], [3, 5], [4, 2], [2, 5], [0, 5]]
    time.sleep(5)
    warehouse.assign_job(0, O[0])
    warehouse.assign_job(1, O[1])
    warehouse.assign_job(2, O[2])
    warehouse.assign_job(1, O[3])
    warehouse.assign_job(0, O[4])

# Main function initializing the warehouse and starting the simulation
def main():
    W, H = 500, 500  # Set the window size as the warehouse size
    Z = [[100, 100], [250, 100], [400, 100], [100, 400], [250, 400], [400, 400]]
    C = [[150, 250], [250, 250], [350, 250]]
    V = [1, 1, 1]  # AGV speeds
    warehouse = Warehouse(W, H, Z, C, V)
    
    agv_thread = threading.Thread(target=AGV_manager, args=(warehouse,), daemon=True)
    agv_thread.start()
    
    warehouse.start()
    pygame.quit()


if __name__ == "__main__": # Entry point
    main()
