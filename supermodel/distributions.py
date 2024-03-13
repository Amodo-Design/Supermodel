if __name__ == "__main__":
    raise Exception("This file is not meant to run by itself, please import it")

#distribution defs taken from modelling utils

from optparse import Option
from typing import TypeVar
import numpy as np
import scipy.stats as stats
from numba.experimental import jitclass
from numba import int32, float64, boolean, float64, int64, boolean, optional
from numba.typed import List
import numpy.typing as npt
from typing import Optional, Union

number_like = Union[int, float, npt.NDArray[np.double]]


class Distribution:
    """Base class for all distributions"""

    weight: float

    def __init__(self):
        if self.weight is None:
            raise ValueError("Must set weight in child class")

    def value_at_x(self, x_value: number_like) -> number_like:
        raise NotImplementedError("Must implement value_at_x method")
    

T = TypeVar('T', bound=Distribution)

spec_cauchy = [
    ('mean', float64),
    ('std_dev', float64),
    ('scale', float64),
    ('y_offset', float64),
    ('x_offset', float64),
    ('input_scale', float64),
    ('positive_allowed', boolean),
    ('negative_allowed', boolean),
    ('skew', float64),
    ('weight', float64),
    ]

#@jitclass(spec_cauchy)
class Distribution_Skewed_Cauchy(Distribution):
    """Skewed Cauchy distribution"""
    def __init__(
        self,
        mean: float, 
        std_dev: float, 
        scale: float, 
        skew: float,
        y_offset: float = 0,
        x_offset: float = 0,
        input_scale: float = 1,
        ):
        
        self.weight: float = 1
        self.mean: float = mean
        self.std_dev: float = std_dev
        self.scale: float = scale
        self.y_offset: float = y_offset
        self.input_scale: float = input_scale
        self.skew: float = skew
        self.x_offset: float = x_offset
        self.y_offset: float = y_offset
        self.positive_allowed: bool = True
        self.negative_allowed: bool = False
        

    def value_at_x(self,x_value: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        x_value = x_value.clip(0,None)
        scaled_x_value = x_value * self.input_scale + self.x_offset
        cauchy_dist_value = self.std_dev / (np.pi * ((scaled_x_value-self.mean)**2) / (self.skew*(scaled_x_value-self.mean) + 1)**2 + self.std_dev**2) + self.y_offset
        output_value = cauchy_dist_value * self.scale
        output_value = np.clip(output_value,a_min=0, a_max=None)

        return output_value
    
spec_normal = [
    ('mean', float64),
    ('std_dev', float64),
    ('scale', float64),
    ('y_offset', float64),
    ('x_offset', float64),
    ('input_scale', float64),
    ('positive_allowed', boolean),
    ('negative_allowed', boolean),
    ('skew', float64),
    ]
#@jitclass(spec_normal)
class Distribution_Normal(Distribution):
    def __init__(self, mean = 0.35, std_dev = 0.35, scale = 8.99E10, y_offset = 0, skew=1, input_scale=1, x_offset=0):
        self.mean = mean
        self.std_dev = std_dev
        self.scale = scale
        self.y_offset = y_offset
        self.skew = skew
        self.positive_allowed = True
        self.negative_allowed = False
        
        self.x_offset = x_offset
        self.input_scale = input_scale

    def value_at_x(self,x_value: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        scaled_x_value = x_value * self.input_scale + self.x_offset

        term_1 = self.std_dev*np.sqrt(2*np.pi)
        term_2 = (scaled_x_value-self.mean)/self.std_dev
        norm_dist_value = (1/term_1) * np.exp(-0.5 * term_2**2)

        fit_value = norm_dist_value * self.scale

        return fit_value

spec_lognorm = [
    ('x_scale', float64),
    ('y_scale', float64),
    ('y_offset', float64),
    ('x_offset', float64),
    ('mean', optional(float64)),
    ('median', optional(float64)),
    ('mode', optional(float64)),
    ('std_dev', float64),
    ('variance', float64),
    ('mu', float64),
    ('sigma', float64),
    ]
#@jitclass(spec_lognorm)
class Distribution_Lognormal(Distribution):
    """Produce a lognormal distribution, with parameters specified as mean, median or mode and standard deviation of the required distribution. Note that this distribution is normalised to have its maximum value at 1, not to be normalised probabilistically."""
    def __init__(
        self,
        x_scale: float = 1,
        y_offset: float = 0,
        x_offset: float = 0,
        mean: Optional[float] =None, 
        median: Optional[float]=None,
        mode: Optional[float] =None,
        y_scale: float=1, 
        *,
        std_dev: float, 
            ):
        
        if mean != None and median != None and mode != None:
            raise ValueError("Must provide only one of mean, median or mode")
        
        self.mean: Optional[float] = mean
        self.median: Optional[float] = median
        self.mode: Optional[float] = mode
        self.std_dev: float = std_dev
        self.variance: float = std_dev ** 2
        self.y_scale: float = y_scale
        self.x_scale: float = x_scale
        self.x_offset: float = x_offset
        self.y_offset: float = y_offset

        #: float64[:] solve for shape parameters of lognormal distribution https://en.wikipedia.org/wiki/Log-normal_distribution

        if self.mean is not None:
            self.sigma = np.sqrt(np.log(1 + (self.variance/self.mean **2)))
            self.mu = np.log(self.mean) - 0.5 * self.sigma ** 2

        elif self.median is not None:
            self.sigma = np.sqrt(np.log((1 + np.sqrt(1 + 4*self.variance * self.median ** -2))/2))
            self.mu = np.log(self.median)
        
        elif self.mode is not None:
            raise NotImplementedError("Mode not implemented yet")
        
        else:
            raise ValueError("Must provide at least one of mean, median or mode")
            
        self.x_offset = self.mu
    
    def value_at_x(self, x_value: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        x_value = x_value * self.x_scale - self.x_offset
        x_value = np.clip(x_value, a_min=0.0001, a_max=None)
        return 1/x_value * np.exp(-(self.sigma ** 2)/2 + self.mu) * np.exp(-((np.log(x_value) - self.mu) ** 2)/(2 * self.sigma ** 2))
        #return stats.lognorm.pdf(x_value + self.mu, self.sigma, loc=self.x_offset, scale=self.y_scale) + self.y_offset


spec_exp = [
    ('b', float64),
    ('scale', float64),
    ('y_offset', float64),
    ('x_offset', float64),
    ('input_scale', float64),
    ('positive_allowed', boolean),
    ('negative_allowed', boolean),
]
#@jitclass(spec_exp)
class Distribution_Exponential(Distribution):
    def __init__(   self, 
                    b:float = 0.2177, 
                    scale:float = 4E-9, 
                    y_offset:float = 0, 
                    ):
        self.scale: float = scale
        self.y_offset: float = y_offset
        self.b: float = b
        self.input_scale:float = 1
        self.x_offset: float = 0
        self.positive_allowed: bool = True
        self.negative_allowed: bool = False
        


    def value_at_x(self,x_value: npt.NDArray[np.double]) -> npt.NDArray[np.double] :
        scaled_x_value = x_value * self.input_scale + self.x_offset

        try:
            exponential_value = self.scale * np.exp(self.b * scaled_x_value) + self.y_offset
        except:
            exponential_value = 0
        if type(exponential_value) is complex:
            exponential_value = 0

        fit_value = np.clip(exponential_value, a_min=0, a_max=None)

        return fit_value

spec_const = [
    ('mean', float64),
    ('weight', float64),
]

#@jitclass(spec_const)
class Distribution_Const(Distribution):
    def __init__(self, mean = 0.35):
        self.mean = mean
        self.weight = 1

    def value_at_x(self,x_value: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        return np.full_like(x_value, fill_value=self.mean)


spec_arh = [
    ('a', float64),
    ('b', float64),
    ('scale', float64),
    ('weight', float64),
    ]
#@jitclass(spec_arh)
class Distribution_Arhennius(Distribution):
    """Used for COVID deactivation rate"""
    def __init__(self, 
                a:float, 
                b:float, 
                scale:float = 1
                ):
        self.a: float = a
        self.b: float = b
        self.scale: float = scale
        self.weight: float = 1
    
    def value_at_x(self,T_value: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        """COVID deactivation rate in 1/s"""
        result = (10 ** ((-self.a/T_value) + self.b))/60
        return result
    
class Distribution_Interpolated(Distribution):
    """Interpolate between some points that are given"""
    def __init__(self, xs: npt.NDArray[np.double], ys:npt.NDArray[np.double], input_scale: float=1, output_scale:float=1):
        self.xs: npt.NDArray[np.double] = xs * input_scale
        self.ys: npt.NDArray[np.double] = ys
        self.output_scale = output_scale

    
    def value_at_x(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.xs, self.ys,) * self.output_scale

spec_pow = [
    ('A', float64),
    ('B', float64),
    ('weight', float64),
    ]
#@jitclass(spec_pow)
class Distribution_Power(Distribution):
    """Power law disrtibution, Ax^B"""
    def __init__(self, A:float, B: float):
        self.A: float = A
        self.B: float = B
        self.weight: float = 1

    def value_at_x(self, x: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        return self.A * x ** self.B
    

def generate_HEPA_filter_dist():
    HEPA_distribution: Distribution_Const = Distribution_Const(
        mean=0.0005,
    )
    return HEPA_distribution

def generate_one_micron_filter_dist():
    one_micron_distribution = Distribution_Interpolated(
        xs=np.array([0,0.99,1.01,2]),
        ys=np.array([1,1,0,0]),
        input_scale=1e-6
    )
    return one_micron_distribution


def generate_perfect_filter_dist():
    filter_distribution: Distribution_Const = Distribution_Const(
    mean = 0,
    )
    return filter_distribution

def generate_zero_dist():
    filter_distribution= Distribution_Const(
    mean = 1
    )
    return filter_distribution