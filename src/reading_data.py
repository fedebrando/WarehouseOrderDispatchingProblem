
import pandas as pd

from dynamic_order import DynamicOrder

def read_data(path: str) -> list[DynamicOrder]:
    '''
    Returns order list by reading file of specified path
    '''
    df = pd.read_csv(path)
    return [DynamicOrder(row.t_arr, row.pick, row.drop) for row in df.itertuples(index=False)]
