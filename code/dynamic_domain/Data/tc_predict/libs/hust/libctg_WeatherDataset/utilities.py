import numpy as np
import math

def RoundBase(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def RangeBase(start, end, step):
    start_range = RoundBase(start, step)
    end_range = RoundBase(end, step) + step
    output = np.arange(start_range, end_range, step, dtype=np.float64).tolist()
    output = [round(value, abs(int(math.log10(step)))) for value in output]
    return output


def CoordsRange(start:float, end:float, step:float, min_val:float, max_val:float) -> list[float]:
    if start < min_val:
        start = min_val
    if end > max_val:
        end = max_val+step
    output = RangeBase(start, end, step)
    return output

def CoordsRangeFromCenter(center:float, dim:int, step:float, min_val:float, max_val:float) -> list[float]:
    start = RoundBase(center, step) - RoundBase(dim/2*step, step)
    end = RoundBase(center, step) + RoundBase(dim/2*step, step)
    output = CoordsRange(start, end, step, min_val, max_val)
    return output

def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return input_list[int(middle)]

def FindNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def GetPrecision(x:float) -> int:
    return round(float(x),5) if len(str(float(x)).split(".")[1]) > 5 else float(x)

def ExtendAxis(axis:np.ndarray, step:float, step_count:int=None, deg_count:float=None):
    if (not step_count and not deg_count):
        raise Exception("Either step_count or deg_count must be given.")
    if (step_count and deg_count):
        raise Exception("Must only give step_count or deg_count.")
    min_value = axis.min()
    max_value = axis.max()
    if not step_count:
        step_count = math.ceil(deg_count/step)
    new_min_value = min_value - step_count * step
    new_max_value = max_value + step_count * step
    new_axis = np.arange(start=new_min_value, stop=new_max_value+(step/2), step=step)
    return new_axis