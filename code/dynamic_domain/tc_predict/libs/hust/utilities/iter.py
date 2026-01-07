from itertools import islice

def IterChunk(arr_range, arr_size) -> list[str]:
    arr_range = iter(arr_range)
    return list(iter(lambda: tuple(islice(arr_range, arr_size)), ()))