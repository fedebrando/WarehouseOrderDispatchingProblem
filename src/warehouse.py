
import pygame
import numpy as np
import math
from state_monitor import StateMonitor
from dynamic_order import DynamicOrder
from stats import Stats
from typing import Callable

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

    def __init__(self, W: int, H: int, Z: list[list[float]], C: list[list[float]], V: list[float]):
        self._init_graphics()
        self._W = W
        self._H = H
        self._state_monitor = StateMonitor(Z, C, V)
        self._running = True
        self._stats = Stats()

    def start(self):
        '''
        Starts the simulation
        '''
        clock = pygame.time.Clock()
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
            self._Z, self._C, self._V, self._O, self._O_ids = self._state_monitor.agv_start()
            self._update()
            self._draw()
            self._state_monitor.agv_end()
            clock.tick(30)  # 30 FPS

    def assign_job(self, policy: Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int], o: DynamicOrder):
        '''
        Assigns a job to an AGV with specified values
        '''
        self._stats.order_arrival(o.get_id())
        self._state_monitor.manager_assign_job(policy, o)
    
    def _init_graphics(self):
        pygame.init()
        self._screen = pygame.display.set_mode((self._WIDTH, self._HEIGHT))  # Define the window here
        pygame.display.set_caption("Warehouse Simulator")
        self._font = pygame.font.Font(None, 24)
    
    def _update(self):
        for idx, c in enumerate(self._C):
            if self._O[idx]:
                dest = self._Z[self._O[idx][0]]
                direction = dest - c
                distance = np.linalg.norm(direction)
                
                if distance > self._V[idx]:
                    shift = (direction / distance) * self._V[idx]
                    self._C[idx] += shift
                else:
                    self._C[idx] = dest
                    print(self._O)
                    self._O[idx].pop(0)
                    if len(self._O[idx]) % 2 == 0: # the order is executed (drop zone has been just removed)
                        self._stats.order_executed(self._O_ids[idx][0])
                        self._O_ids[idx].pop(0)
                        print(self._stats.mean_waiting_time())
            
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
