
class DynamicOrder:
    '''
    Dynamic order
    '''
    _curr_id = 1

    def __init__(self, t_arr: float, pick: int, drop: int):
        self._id = DynamicOrder._curr_id
        DynamicOrder._curr_id += 1
        self._t_arr = t_arr
        self._pick = pick
        self._drop = drop

    def get_t_arr(self) -> float:
        '''
        Returns the arrival time
        '''
        return self._t_arr
    
    def get_pick(self) -> int:
        '''
        Returns the pick zone
        '''
        return self._pick
    
    def get_drop(self) -> int:
        '''
        Returns the drop zone
        '''
        return self._drop
    
    def __repr__(self) -> str:
        return f'({self._id}, {self._t_arr}, {self._pick} -> {self._drop})'
