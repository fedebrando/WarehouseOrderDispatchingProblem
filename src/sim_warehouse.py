
import pygame
import numpy as np
import math
from typing import Callable

from sim_state_monitor import StateMonitor
from dynamic_order import DynamicOrder
from sim_stats import Stats

class Warehouse:
    '''
    Warehouse Simulator
    '''
    _FPS = 30

    _SHORT_DELAY = (255, 153, 0)
    _MID_DELAY = (255, 0, 0)
    _STRONG_DELAY = (153, 0, 255)

    _GRID_COLOR = (50, 50, 50)
    _WIDTH_TERMINAL = 500
    _TERMINAL_DEFAULT_COLOR = (255, 255, 255)
    _TERMINAL_HEADER_COLOR = (255, 215, 0)
    _BG_COLOR = (30, 30, 30)
    _ZONE_COLOR = (100, 100, 100)
    _ZONE_TEXT_COLOR = (175, 175, 175)
    _AGV_TEXT_COLOR = (30, 30, 30)
    _AGV_COLOR_IDLE = (0, 255, 0)
    _AGV_COLOR_BUSY = (255, 0, 0)

    def __init__(self, name: str, Z: list[list[float]], C: list[list[float]], V: list[float], P: float, th_short: float = 0, th_mid: float = 5, th_strong: float = 10):
        self._name = name

        all_pos = Z + C
        self._W = max(x for [x, _] in all_pos) + min(x for [x, _] in all_pos)
        self._H = max(y for [_, y] in all_pos) + min(y for [_, y] in all_pos)
        self._init_graphics()
        self._state_monitor = StateMonitor(Z, C, V, P)
        self._running = True

        # Statistics
        self._stats = Stats()

        # Delay thresholds
        self._th_short = th_short
        self._th_mid = th_mid
        self._th_strong = th_strong

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
            clock.tick(self._FPS)

    def assign_job(self, policy: Callable[[DynamicOrder, np.array, np.array, np.array, list[list[int]]], int], o: DynamicOrder, time_diff: float):
        '''
        Assigns a job to an AGV with specified values
        '''
        self._stats.order_arrival(o.get_id())
        self._stats.new_time_diff(o.get_id(), time_diff)
        self._state_monitor.manager_assign_job(policy, o, self._stats)
    
    def _init_graphics(self):
        '''
        Initialize pygame graphics
        '''
        pygame.init()
        self._screen = pygame.display.set_mode((self._W + self._WIDTH_TERMINAL, self._H))  
        pygame.display.set_caption(self._name)
        self._font = pygame.font.SysFont('Courier', 18)
    
    def _delay_color(self, delay: float):
        '''
        Matches delay with relative color
        '''
        if delay > self._th_strong:
            return self._STRONG_DELAY
        if delay > self._th_mid:
            return self._MID_DELAY
        if delay > self._th_short:
            return self._SHORT_DELAY
        return self._TERMINAL_DEFAULT_COLOR

    def _generate_terminal_data(self):
        '''
        Generates a structured table of AGV data for the terminal display
        '''
        header = f"{'AGV':<5}{'Delay (s)':<11}{'Order ID':<10}{'Path'}"
        terminal_data = [(header, self._TERMINAL_HEADER_COLOR)]
        
        for idx, c in enumerate(self._C):
            assigned_order = f'{self._O_ids[idx][0]}' if self._O_ids[idx] else '-'
            path = ' > '.join(map(self._number_to_excel_column, self._O[idx])) if self._O[idx] else '-'
            d = self._stats.delay_of(self._O_ids[idx][0]) if self._O_ids[idx] else 0
            delay = f'{d:.2f}' if d > 0 else '-'
            terminal_data.append((f"{str(idx + 1):<5}{f'{delay}':<11}{assigned_order:<10}{path}", self._delay_color(d)))

        terminal_data.append(('', self._TERMINAL_DEFAULT_COLOR))

        terminal_data.append((f"{'Mean waiting time':<20}{self._stats.mean_waiting_time():.7g} s", self._TERMINAL_DEFAULT_COLOR))
        terminal_data.append((f"{'Mean distance':<20}{self._stats.mean_distance():.7g} m", self._TERMINAL_DEFAULT_COLOR))
        terminal_data.append((f"{'Mean consumption':<20}{self._stats.mean_consumption():.7g} J", self._TERMINAL_DEFAULT_COLOR))
        
        return terminal_data

    def _update(self):
        '''
        Updates data
        '''
        for idx, c in enumerate(self._C):
            if self._O[idx]:
                dest = self._Z[self._O[idx][0]]
                direction = dest - c
                distance = np.linalg.norm(direction)
                
                ds = self._V[idx] * (1 / self._FPS)

                if distance > ds:
                    shift = (direction / distance) * ds
                    self._C[idx] += shift
                else:
                    self._C[idx] = dest
                    self._O[idx].pop(0)
                    if len(self._O[idx]) % 2 == 0:  # The order is executed (drop zone just removed)
                        self._stats.order_executed(self._O_ids[idx][0])
                        self._O_ids[idx].pop(0)

    def _draw_grid(self):
        '''
        Draws a grid on the warehouse floor
        '''
        grid_size = 50  # Size of each grid square
        for x in range(0, self._W, grid_size):
            pygame.draw.line(self._screen, self._GRID_COLOR, (x, 0), (x, self._H))
        for y in range(0, self._H, grid_size):
            pygame.draw.line(self._screen, self._GRID_COLOR, (0, y), (self._W, y))

    def _draw_terminal(self):
        '''
        Draws a structured terminal-style display in the top-right corner
        '''
        x, y = self._W, 0  # Position in the top-right corner

        # Draw a semi-transparent rectangle
        terminal_surface = pygame.Surface((self._WIDTH_TERMINAL, self._H), pygame.SRCALPHA)
        terminal_surface.fill((0, 0, 0, 200))  # Semi-transparent black
        self._screen.blit(terminal_surface, (x, y))

        # Get terminal data and render it
        terminal_data = self._generate_terminal_data()
        
        for i, (message, color) in enumerate(terminal_data):
            label = self._font.render(message, True, color)
            self._screen.blit(label, (x + 10, y + 10 + i * 20))  # Offset each line

    def _number_to_excel_column(self, n: int) -> str:
        '''
        Converts a number into a Excel-like column header string
        '''
        result = ''
        n += 1
        while n > 0:
            n -= 1
            result = chr(n % 26 + 65) + result
            n //= 26
        return result

    def _draw(self):
        '''
        Draws a single frame
        '''
        self._screen.fill(self._BG_COLOR)
        
        # Draw the grid
        self._draw_grid()
        
        # Draw zones with numbers
        for idx, z in enumerate(self._Z):
            pygame.draw.rect(self._screen, self._ZONE_COLOR, (int(z[0]) - 10, int(z[1]) - 10, 20, 20))
            
            # Display only the zone number
            label = self._font.render(self._number_to_excel_column(idx), True, self._ZONE_TEXT_COLOR)
            x_pos = int(z[0] - label.get_width() / 2)
            y_pos = int(z[1] - label.get_height() / 2)
            self._screen.blit(label, (x_pos, y_pos))
        
        # Draw AGVs and arrows
        for idx, c in enumerate(self._C):
            color = self._AGV_COLOR_BUSY if self._O[idx] else self._AGV_COLOR_IDLE

            # Draw direction arrow if AGV has an order
            if self._O[idx]:
                destination = self._Z[self._O[idx][0]]
                angle = math.atan2(destination[1] - c[1], destination[0] - c[0])

                arrow_length = 20
                arrow_width = 5
                arrow_points = [
                    (c[0] + math.cos(angle) * arrow_length, c[1] + math.sin(angle) * arrow_length),
                    (c[0] + math.cos(angle + math.pi / 6) * arrow_width, c[1] + math.sin(angle + math.pi / 6) * arrow_width),
                    (c[0] + math.cos(angle - math.pi / 6) * arrow_width, c[1] + math.sin(angle - math.pi / 6) * arrow_width),
                ]
                pygame.draw.polygon(self._screen, color, arrow_points)

            # Draw the AGV
            pygame.draw.circle(self._screen, color, (int(c[0]), int(c[1])), 10)
            label = self._font.render(f'{idx + 1}', True, self._AGV_TEXT_COLOR)
            x_pos = int(c[0] - label.get_width() / 2)
            y_pos = int(c[1] - label.get_height() / 2)
            self._screen.blit(label, (x_pos, y_pos))

        # Draw the terminal box
        self._draw_terminal()

        pygame.display.flip()
