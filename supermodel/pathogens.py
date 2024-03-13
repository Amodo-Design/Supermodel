"""
Classes, relating to pathogens
"""

if __name__ == "__main__":
    raise EnvironmentError("This file is not meant to ran as main. Please import it into another file.")

from . import distributions, classes
import tomli
import numpy.typing as npt
import numpy as np
from typing import Union, Optional
from supermodel import CONFIG_PATH

number_like = Union[int, float, npt.NDArray[np.double]]

class Pathogen:

    def __init__(self,
        # take all paramaters that are in TOML as arguments
        name: str,
        infectivity_curve: distributions.Distribution,
        liquid_deactivation_curve: distributions.Distribution,
        cp: float, # Concentration of infectant bodies in droplets. count/ml
        pi: float, # Infectivity of infectant bodies: infection chance per count
        pathogen_size: float, # in nm

        time_to_symptoms: float, # Time to symptoms in seconds
        time_to_recovery: float, # Time to recovery in seconds
        time_to_death: float, # Time to death in seconds
        lethality: float, # Fraction of infected individuals who die

        # set paramaters not in spreadsheet with default values
        UV_deactivation_rates : dict = {}, # UV deactivation rate in cm^2/mJ
        asymptomatic_proportion :float = 0.5, # proportion of infections that are asymptomatic
        recovery_immunity :float = 1.0, # Proportion of Recovered for which recovery grants immunity
        droplet_viscosity :float = 1.5E-3, # Typical pathogen-containing-droplet viscosity in Pa s
        droplet_density: float = 1000 ,# Typical pathogen-containing-droplet density in kg/m3
        populations: Optional[dict] = None # Dictionary of populations of this pathogen
    ):

        if populations is None:
            self.populations = {"default": {}}
            self.populations['default'].update(UV_deactivation_rates)
            self.populations['default'].update({"population_fraction": 1.0})
            
        else: 
            self.populations = populations

        self.name = name
        self.infectivity_curve = infectivity_curve
        self.cp = cp 
        self.pi = pi 
        self.phi = self.cp * self.pi # (infection chance/m3 fluid)
        self.pathogen_size = pathogen_size

        self.time_to_symptoms = time_to_symptoms 
        self.time_to_death = time_to_death 
        self.time_to_recovery = time_to_recovery
        self.lethality = lethality

        self.asymptomatic_proportion = asymptomatic_proportion
        self.recovery_immunity = recovery_immunity
        self.droplet_viscosity = droplet_viscosity
        self.droplet_density = droplet_density

        self.liquid_deactivation_curve = liquid_deactivation_curve
        

    def deathFunc(self, time_since_infection: float) -> float:
        """Gives the probability of death of infected person given time since infection"""
        ##currently just a simple uniform distribution between time_to_death and time_to_recovery
        if time_since_infection < self.time_to_death:
            return 0.0
        else:
            return self.lethality/(self.time_to_recovery - self.time_to_death)

    def deactivation_rate_liquid(self,temperature: number_like) -> number_like:
        return self.liquid_deactivation_curve.value_at_x(temperature)

    @property
    def population_count(self) -> int:
        """Return the number of populations of this pathogen"""
        return len(self.populations)
    
    
    def infectivity_prototype(self, radii: npt.NDArray[np.double]):
        return classes.infectivity([(name, np.zeros_like(radii)) for name, data in self.populations.items()])
    
    def UV_decay(self, fluence, wavelength):
        """
        Return a dictionary of decay rates for each population of this pathogen, in an infectivity object, ready to be multiplied by an infectivity object
        """
        if not wavelength:
            return classes.infectivity([(name, np.zeros_like(fluence)) for name, data in self.populations.items()])
        return classes.infectivity([
            (name, data[str(wavelength)] * fluence) for name, data in self.populations.items()
        ])


"""
Generate pathogen related data
"""

if CONFIG_PATH is not None:
    with open(CONFIG_PATH, 'rb') as setup_file:
        setup_params = tomli.load(setup_file)
else:
    raise Exception('CONFIG_PATH not set, please run supermodel.loadconfig(path)')
def read_pathogen_data():
    """Return a Pandas dataframe containing pathogen data"""
    pathogen_data = tomli.load(setup_params["pathogen_data"])
    return pathogen_data

def generate_covid_19_liq_deactivation_curve():
    """Return deactivation curve for covid-19 in liquid given a temperature in Kelvin"""
    deactivation_curve = distributions.Distribution_Arhennius(
        a = 5252.3,
        b = 15.039
    )
    return deactivation_curve
