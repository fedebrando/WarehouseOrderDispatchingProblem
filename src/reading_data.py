
import pandas as pd
from dynamic_order import DynamicOrder

def read_data(path: str) -> list[DynamicOrder]:
    df = pd.read_csv(path)
    return [DynamicOrder(row.t_arr, row.pick, row.drop) for row in df.itertuples(index=False)]
