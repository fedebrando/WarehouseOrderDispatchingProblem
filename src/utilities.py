
def path_length(*path: tuple[float, float]) -> float:
    '''
    Returns length of the received path
    '''
    path = list(path)
    return sum(d(path[i], path[i+1]) for i in range(len(path) - 1))

def d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    '''
    Returns squared distance between two point
    '''
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
