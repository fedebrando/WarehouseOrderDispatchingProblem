import csv
import random

NUM_ORDERS = {
    'train': 7000,
    'val': 1500,
    'test': 3500
}

# λ chosen to get an average inter-arrival time close to 50
LAMBDA_RATE = 1 / 50  # i.e., average time ≈ 50

def generate_orders(filename: str, num_orders: int):
    '''Generates a CSV file with random orders, ensuring increasing arrival times using exponential distribution.'''
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['t_arr', 'pick', 'drop'])
        
        arrival_time = 0
        for _ in range(num_orders):
            # Exponential inter-arrival time
            arrival_time += random.expovariate(LAMBDA_RATE)
            loading_zone = random.randint(0, 5)
            unloading_zone = random.choice([n for n in range(6) if n != loading_zone])
            writer.writerow([round(arrival_time, 2), loading_zone, unloading_zone])

def main():
    random.seed(42)  # seed for reproducibility (optional)
    for subset, num_orders in NUM_ORDERS.items():
        filename = f'orders_{subset}.csv'
        generate_orders(filename, num_orders)
        print(f'File {filename} generated successfully')

if __name__ == '__main__':
    main()
