# Some custom classes and functions for the supermodel package, used internally by the package.

import numpy.typing as npt
import numpy as np
import json

class infectivity(dict):
    """
    Class to hold infectivity data. Inherits from dict, so can be used as a dict, but also has some extra mathematical functionality.
    """
    def __mul__(self, other):
        if isinstance(other, dict):
            self.__checkkeys__(other)
            
            return infectivity([(key, data * other[key]) for key, data in self.items()])
        return infectivity([(key, data * other) for key, data in self.items()])
    
    def __imul__(self, other):
        if isinstance(other, dict):
            self.__checkkeys__(other)
            
            self.update([(key, data * other[key]) for key, data in self.items()])
            return self
        self.update([(key, data * other) for key, data in self.items()])
        return self
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if isinstance(other, dict):
            self.__checkkeys__(other)
            
            return infectivity([(key, data + other[key]) for key, data in self.items()])
        
        return infectivity([(key, data + other) for key, data in self.items()])
    
    def __iadd__(self, other):
        if isinstance(other, dict):
            self.__checkkeys__(other)
            
            self.update([(key, data + other[key]) for key, data in self.items()])
            return self
        self.update([(key, data + other) for key, data in self.items()])
        return self
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __array__(self, dtype=None):
        return np.array(list(self.values()), dtype=dtype)
    
    def sum(self):
        return np.sum(self.__array__())
    
    def __sub__(self, other):
        if isinstance(other, dict):
            self.__checkkeys__(other)
            return infectivity([(key, data - other[key]) for key, data in self.items()])
        return infectivity([(key, data - other) for key, data in self.items()])
    
    def __rsub__(self, other):
        if isinstance(other, dict):
            self.__checkkeys__(other)
            return infectivity([(key, other[key] - data) for key, data in self.items()])
        return infectivity([(key, other - data) for key, data in self.items()])
    
    def __isub__(self, other):
        self.update([(key, data - other) for key, data in self.items()])
        return self
    
    def __checkkeys__(self, other):
        if set(self.keys()) != set(other.keys()):
            raise ValueError('Cannot divide infectivity objects with different keys')
        
class superJSONencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



def convert_to_seconds(unit: str, value: float) -> float:
    """Convert a list of times to seconds based on their units - case sensitive.
    Unit must be one of the strings:
    - days
    - hours
    - minutes
    - seconds
    """

    match unit:
        case "days":
            multiplier = 24*3600

        case "hours":
            multiplier = 3600

        case "minutes":
            multiplier = 60

        case "seconds":
            multiplier = 1
        case _:
            raise ValueError("Unit not recognized: {}".format(unit))

    return value * multiplier