"""
Classes, relating to all things people
"""
from __future__ import annotations

from pyparsing import Opt
if __name__ == "__main__":
    raise EnvironmentError("This file is not meant to run by itself, please import it")
from optparse import Option
import numpy as np
from . import distributions, interventions, pathogens
import random
import copy
from numba import jit, int32, float64, boolean, float64, int64, boolean, optional, char, types, typed
from numba.experimental import jitclass
from typing import Optional, List
import numpy.typing as npt

class Occupant:

    #totals for all occupants, updated by setters 

    def __init__(self,
        #Basic paramaters that have no default value and must be defined on instance creation
        activity_distributions: List[distributions.Distribution],
        pathogen: pathogens.Pathogen,
        droplet_radii: npt.NDArray[np.double],
        quarantine_compliance: float, # Effectivity of quarantine upon emergence of symptoms
        breathing_flowrate: float,
        ppe_compliance: float,
        ppe_fit: float, # 1 - fitment of PPE, 0 = perfect fit, 1 = no fit
        respiratory_ppe: Optional[List[interventions.PPE]],
        unique_details: Optional[dict] = None,
    ):
        """The class that holds occupant data thoughout the simulation, including methods to get the viral in/out flows for occupants and changing their states. 
        
        Essentially a state machine capable of calculating simulation variables"""

        #take variables from init call
        self.breathing_flowrate = breathing_flowrate
        self._respiratory_ppe = respiratory_ppe
        self.pathogen = pathogen

        self.activity_distributions = activity_distributions
        self.quarantine_compliance = quarantine_compliance
        self.ppe_compliance = ppe_compliance
        self._ppe_fit = ppe_fit

        #Default bools for occupant whom is susceptible. Interact with non-underscore @properties
        self._susceptible: bool = True
        self._infected = False
        self._recovered = False
        self._is_symptomatic = False
        self._quarantined = False
        self._deceased = False

        self._infectious = True

        #set random variables (compliances)
        self.recalc_random()

        #Uninitialised variables updated by some methods used at simulation time
        self.time_infected: float = np.inf #time occupant was infected, will be set by self.infect()
        self._outgoing_viral_load: Optional[float] = None
        self._incoming_viral_load: Optional[float] = None

        #reducing num function calls (caching optimisation)
        self._cache_droplet_radii = droplet_radii
        self._cache_zeros_like_radii = np.zeros_like(self._cache_droplet_radii)
        
        self._recache_resp_ppe()

        self._cache_total_droplet_distribution_weight = np.sum([_distribution.weight for _distribution in self.activity_distributions]) #get total weight of multiple droplet distributions
        self._cache_activity_dist = np.sum([_distribution.weight * _distribution.value_at_x(self._cache_droplet_radii)/self._cache_total_droplet_distribution_weight for _distribution in self.activity_distributions], axis=0)# create combination of multiple droplet distributions

        ##deal with totals
        self.totals: dict[str, int]

        #Occupant specific
        self.details = unique_details


    def __str__(self) -> str:
        return (
            f"Occupant: \n" 
            f"{self.susceptible=} \n" 
            f"{self.infected=} \n" 
            f"{self.recovered=} \n" 
            f"{self.deceased=} \n" 
            f"{self.quarantined=} \n" 
            f"{self.is_symptomatic=} \n" 
            f"{self.infectious=} \n" 
            f"{self.details=} \n"
        )

    def __deepcopy__( self, memo): #occupants need to recalc random variables when copied, and preserve totals
        cls = self.__class__
        result = memo.get(id(self))
        if result is not None:
            return result
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        self.recalc_random()
        return result

    def recalc_random(self):
        """Recalculate random variables for occupant, used when occupant is reinitialised"""
        self.comply_quarantine = random.random() < self.quarantine_compliance
        self.comply_PPE = random.random() < self.ppe_compliance

    @property
    def ppe_fit(self):
        return self._ppe_fit
    
    @ppe_fit.setter
    def ppe_fit(self, value):
        self._ppe_fit = value
        self._recache_resp_ppe()

    @property
    def respiratory_ppe(self) -> Optional[List[interventions.PPE]]:
        return self._respiratory_ppe
    
    @respiratory_ppe.setter
    def respiratory_ppe(self, value):
        self._respiratory_ppe = value
        self._recache_resp_ppe()

    def _recache_resp_ppe(self):
        """Useful if manually setting the PPE for occupants, called by respiratory_ppe.setter"""
        if self.respiratory_ppe is not None:
            _resp_PPE_in_curve = np.array([_ppe_curve.resp_pen_in_curve.value_at_x(self._cache_droplet_radii) for _ppe_curve in self.respiratory_ppe]).prod(axis=0)
            _resp_PPE_out_curve = np.array([_ppe_curve.resp_pen_out_curve.value_at_x(self._cache_droplet_radii) for _ppe_curve in self.respiratory_ppe]).prod(axis=0)

            if not self.comply_PPE:
                self._resp_PPE_in_curve = np.ones_like(_resp_PPE_in_curve)
                self._resp_PPE_out_curve = np.ones_like(_resp_PPE_out_curve)
            else:
                self._resp_PPE_in_curve = _resp_PPE_in_curve
                self._resp_PPE_out_curve = _resp_PPE_out_curve

        else: 
            self._resp_PPE_out_curve = np.full_like(self._cache_droplet_radii,1)
            self._resp_PPE_in_curve = np.full_like(self._cache_droplet_radii,1)

    def calculate_outgoing(self, time: float) -> npt.NDArray[np.double]:
        """Calculate the outgoing infectvity of occupant for a given radius and time (time important due to infectivity curves)"""

        self._recache_resp_ppe()

        #return zero if occupant not infected (only for optimisation)
        if not self.infected or self.deceased or self.quarantined or self.susceptible: 
            return self._cache_zeros_like_radii

        else:
            #infctivity curve value at time since infection
            infectivity_multiplier = self.pathogen.infectivity_curve.value_at_x(time - self.time_infected)
            #droplet number distribution given respiratory activity per unit volume breathed 
            droplet_density_curve = self._cache_activity_dist
            #breathing flowrate per unit time
            breath_rate = self.breathing_flowrate
            #l outgoing PPE penetration for droplet size
            PPE_penetration = self._resp_PPE_out_curve

            #outgoing infectivity for droplet size for this occupant PER UNIT TIME PER RADIUS
            return PPE_penetration *  infectivity_multiplier * droplet_density_curve * breath_rate * self.infectious

    def incoming_factors(self) -> npt.NDArray[np.double]:
        """Calculate the incoming infectivity multiplicative factor of the occupant for each sample radius, accounting for PPE PER UNIT TIME PER RADIUS"""
        if self.infected or not self.susceptible or self.deceased or self.quarantined:
            return self._cache_zeros_like_radii
        
        else:
            PPE_penetration = self._resp_PPE_in_curve
            return self.breathing_flowrate * self.ppe_compliance * self.susceptible * PPE_penetration
    
    def kill(self) -> None:
        """Kill the occupant, setting all state variables to false and deceased to true"""
        self.deceased = True
        self.susceptible = False
        self.infected = False
        self.quarantined = False

    def infect(self, time: float, pathogen:pathogens.Pathogen) -> None:
        """Infect the occupant
        time: time of infection in seconds
        pathogen: pathogen to infect occupant with"""
        self.pathogen = pathogen
        self.infected = True
        self.susceptible = False
        self.time_infected = time
        if random.random() > self.pathogen.asymptomatic_proportion: self.is_symptomatic = True

    def quarantine(self) -> None:
        """Quarantine the occupant"""
        self.quarantined = True

    def recover(self) -> None:
        """Recover the occupant"""
        self.recovered = True
        self.quarantined = False
        self.infected = False

    #property getters and setters for summed variables (in Occupant.totals)
    @property
    def susceptible(self):
        return self._susceptible
    
    @susceptible.setter
    def susceptible(self, value) -> None:
        self.totals['susceptible'] += value - self._susceptible
        self._susceptible = bool(value)

    @property
    def infected(self) -> bool:
        return self._infected
    @infected.setter
    def infected(self, value: bool) -> None:
        self.totals['infected'] += value - self._infected
        self._infected = bool(value)

    @property
    def recovered(self) -> bool:
        return self._recovered
    
    @recovered.setter
    def recovered(self, value:bool) -> None:
        self.totals['recovered'] += value - self._recovered
        self._recovered = bool(value)

    @property
    def quarantined(self) -> bool:
        return self._quarantined
    @quarantined.setter
    def quarantined(self, value: bool) -> None:
        self.totals['quarantined'] += value - self._quarantined
        self._quarantined = bool(value)

    @property
    def is_symptomatic(self) -> bool:
        return self._is_symptomatic
    @is_symptomatic.setter
    def is_symptomatic(self, value: bool) -> None:
        self._is_symptomatic = bool(value)

    @property
    def deceased(self) -> bool:
        return self._deceased
    @deceased.setter
    def deceased(self, value: bool) -> None:
        self.totals['deceased'] += value - self._deceased
        self._deceased = bool(value)

    @property
    def infectious(self) -> bool:
        return self._infectious
    @infectious.setter
    def infectious(self, value: bool) -> None:
        self._infectious = bool(value)
"""
Generate human related distributions
"""


def generate_quiet_speech_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.33,
    std_dev = 0.2,
    scale = 5.34E10,
    skew = 1.0,
    y_offset = -0.1,
    input_scale = 1E6,
    )
    return speech_dist

def generate_quiet_speech_half_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.31,
    std_dev = 0.18,
    scale = 4.67E10,
    skew = 1.0,
    y_offset = -0.1,
    input_scale = 1E6,
    )
    return speech_dist

def generate_breathing_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.29,
    std_dev = 0.16,
    scale = 4.0E10,
    skew = 1.0,
    y_offset = -0.1,
    input_scale = 1E6,
    x_offset=0,
    )
    return speech_dist

def generate_heavy_breathing_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.29,
    std_dev = 0.16,
    scale = 8.0E10,
    skew = 1.0,
    y_offset = -0.1,
    input_scale = 1E6
    )
    return speech_dist

def generate_intermediate_speech_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.35,
    std_dev = 0.35,
    scale = 1.80E11,
    skew = 1.0,
    y_offset = -0.2,
    input_scale = 1E6,
    )
    return speech_dist

def generate_intermittent_shouting_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.35,
    std_dev = 0.35,
    scale = 1.80E11*1.5,
    skew = 1.0,
    y_offset = -0.2,
    input_scale = 1E6,
    )
    return speech_dist

def generate_loud_singing_dist(weight:float):
    speech_dist = distributions.Distribution_Skewed_Cauchy(
    mean = 0.485,
    std_dev = 0.40,
    scale = 9.54E11,
    skew = 1.0,
    y_offset = -0.2,
    input_scale = 1E6,
    )
    return speech_dist


