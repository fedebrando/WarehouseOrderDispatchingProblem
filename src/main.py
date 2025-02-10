
import time
import threading
from warehouse import Warehouse
import pygame

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
