"""
Main model class and methods
"""
if __name__ == "__main__":
    raise EnvironmentError("This module is not intended to be ran from the command line")

from . import environments, humans, pathogens, classes
from .classes import convert_to_seconds
import pandas as pd
import numpy as np
import random
import scipy.constants as const
from rich.progress import Progress, TextColumn, TaskID
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from rich import print
import copy
import tomli
from numba import jit, float64
from typing import Optional, Union
import numpy.typing as npt
from supermodel import CONFIG_PATH

if CONFIG_PATH is not None:
    with open(CONFIG_PATH, 'rb') as setup_file:
        setup_params = tomli.load(setup_file)
else:
    raise Exception('CONFIG_PATH not set, please run supermodel.loadconfig(path)')

class Model:

    def __init__(self,
                                  
        #must be defined at class init
        room: environments.Room, 
        occupants: list[humans.Occupant],
        pathogen: pathogens.Pathogen,
        droplet_radii:npt.NDArray[np.double],
        droplet_radius_step:float,
        model_break_continue_times: Optional[npt.NDArray[np.double]],
        iszonal: bool,
        UV_type: Optional[str],
                 
        #defaults overidden by genreate_case_study
        time_step_size: float  = 20,
        max_time: float = 30 * 24 * 3600,
        quarantine = False, #whether quarantine is enabled

        #model data
        time_data: Optional[npt.NDArray[np.double]] = None,
        infection_data: Optional[npt.NDArray[np.double]] = None,
        name: Optional[str] = None,
    ):
        """
        The class that holds the modelling methods used for a simulation. Returned by parse_dataframe_case_studies() and used by Supermodel().
        
        Zonal models:
            The model stores the different pathogen concentratoins in upper and lower zones in self._total_infectivity which is a 2-tuple. The 0th entry is the lower zone concentration and the 1st entry is the upper zone entry.
        """
        self.verbose = False

        self.iszonal = iszonal
        self.UV_type = UV_type
        self.droplet_radii = droplet_radii
        self.droplet_radius_step = droplet_radius_step
        self.occupants = occupants
        self.room: environments.Room = room
        self.time_step_size = time_step_size
        self.pathogen: pathogens.Pathogen = pathogen
        self.max_time = max_time
        self.model_break_continue_times = model_break_continue_times
        self.critical_radius = calculate_critical_radius(self.pathogen, self.room)

        #list of column names for dataframes
        self.data_headers = [
                    "Susceptible Occupants",
                    "Infected Occupants",
                    "Infected - Not Quarantined Occupants",
                    "Dead Occupants",
                    "Quarantined Occupants",
                    "Recovered Occupants",
                    "Viral Load - Lower",
                    "Viral Load - Upper",
                    "Time"
                    ] 
        #empty list to hold data
        self.data = [] * len(self.data_headers)

        #start room at zero infectivity for all radii
        self._total_infectivity = {
            'upper' : self.pathogen.infectivity_prototype(self.droplet_radii),
            'lower' : self.pathogen.infectivity_prototype(self.droplet_radii)
        }

        self.time: float = 0

        #Cache some variables for speed
        self._cache_critical_radius = calculate_critical_radius(self.pathogen, self.room)
        self._cache_phi = self.pathogen.phi
        self._cache_Vd = calculate_droplet_volume(self.droplet_radii)
        
        #Get data from case study
        self.time_data = time_data
        self.infection_data = infection_data
        self.name= name

        #Cache some variables
        self._cache_pathogen_deactivation_rate = self.pathogen.deactivation_rate_liquid(self.room.temperature)

        #do quarantine check - if not enables set all occupant quarantine compliance to 0
        if not quarantine: 
            for _occupant in self.occupants: _occupant.quarantine_compliance = 0

        if self.name is not None: 
            print("[green][bold]Initialised model " + self.name)
        else: 
            print("[green][bold]Initialised custom model...")

        #construct totals dict at initialisation
        self.totals = {
            'susceptible': sum(occupant.susceptible for occupant in self.occupants),
            'infected': sum(occupant.infected for occupant in self.occupants),
            'recovered': sum(occupant.recovered for occupant in self.occupants),
            'quarantined': sum(occupant.quarantined for occupant in self.occupants),
            'deceased': sum(occupant.deceased for occupant in self.occupants),
            }

    def __str__(self) -> str:
        _output= (  "[cyan][bold italic]An epidemic model in the supermodel package \n[/]" 
                    + "[green][bold]NAME[/bold]: {} \n".format(self.name) 
                    + "\t time step size: {} \n".format(self.time_step_size) 
                    + "\t max time: {} \n".format(self.max_time) 
                    + "\t break continue times: {} \n".format(self.model_break_continue_times) 
                    + "[red][bold]PATHOGEN[/bold]: {} \n".format(self.pathogen.name) 
                    + "\t pi: {} \n".format(self.pathogen.pi) 
                    + "\t cp: {} \n".format(self.pathogen.cp) 
                    + "[blue][bold]ROOM[/bold]:\n"
                    + "\t height: {} \n".format(self.room.height)
                    + "\t volume: {} \n".format(self.room.volume)
                    + "\t external air change rate: {} \n".format(self.room.external_air_change_rate)
                    + "[cyan][bold]OCCUPANTS:[/bold] \n" 
                    + "\t {} total occupants \n".format(len(self.occupants)) 
                    + "\t {} infected occupants \n".format(self.count_infected()) 
                    + "\t {} susceptible occupants \n".format(self.count_susceptible()) 
                    + "\t {} quarantined occupants \n".format(self.count_quarantined()) 
                    + "\t {} recovered occupants \n".format(self.count_recovered()) 
                    )
        return _output
    
    def count_susceptible(self) -> int:
        """Return number of susceptile occupants"""
        return self.totals['susceptible']

    def count_infected(self) -> int:
        """Return number of infected occupants"""
        return self.totals['infected'] 
    
    def count_infected_not_quarantined(self):
        """Return number of infected occupants that have not been quarantined"""
        return self.count_infected() - self.count_quarantined()
    
    def count_dead(self):
        """Return number of dead occupants"""
        return self.totals['deceased'] 
    
    def count_quarantined(self):
        """Return number of quarantined occupants"""
        return self.totals['quarantined']
    
    def count_recovered(self):
        """Return number of recovered occupants"""
        return self.totals['recovered']
    
    def get_total_infectivity_step(self):
        """The total outbound infectivity of all occupants for each radius, sum rates over occupants and multiply by timestep"""
        increment =  np.sum([_occupant.calculate_outgoing(time=self.time) for _occupant in self.occupants], axis=0) \
            * self.time_step_size \
            * self.droplet_radius_step \
            * self._cache_phi \
            * self._cache_Vd 
        return increment
    
    def checksmallfactor(self, factor: Union[float, npt.NDArray], threshold: float = 0.05):
        """Check if a factor is small, and if not, raise error"""
        if isinstance(factor, float): 
            if factor > threshold: raise ValueError("Factor {:.2E} is not small enough for this calculation (threshold {:.2E})".format(factor, threshold))
        else:
            if np.any(factor > threshold): raise ValueError("Factor {:.2E} is not small enough for this calculation (threshold {:.2E})".format(factor, threshold))

    def new_infectivity(self, total_infectivity: npt.NDArray[np.double]) -> classes.infectivity:
        """
        Return new infectivity to be added to the total, according to the pathogen's population ratios
        """
        return classes.infectivity([(name, total_infectivity * data["population_fraction"]) for name, data in self.pathogen.populations.items()])

    def run(self, progress: Optional[Progress], task_id: Optional[TaskID]):
        """Run simulation steps and store data
        progress: Rich progress bar object
        task_id: Rich progress bar task id"""


        def simulate_step(occupant_freeze:bool, end_time: float):
            """Run the simulation"""
            nonlocal progress
            nonlocal task_id
            
            while self.time < end_time:
                #add infectivity from occupants to room (per radius per volume)
                if not occupant_freeze: 
                    self._total_infectivity['lower'] += self.new_infectivity(self.get_total_infectivity_step())  
                    self.occupants = [self.occupant_update(_occupant) for _occupant in self.occupants] 
                #calculate total decay rate for lower zone
                _lower_volumetric_decay_rate = (self.room.lower_zone.external_air_outflow_rate * self.room.volume\
                                            - self.room.net_flow_rate * self.room.net_flow_outflow_fraction * self.room.volume \
                                            + self.room.lower_zone.filtration_air_change_rate_roomvol_hourly * self.room.volume \
                                            + self._cache_pathogen_deactivation_rate) /3600 \
                                            / self.room.lower_zone.volume

                self.checksmallfactor(_lower_volumetric_decay_rate * self.time_step_size)
                _lower_UV_decay_rate = self.pathogen.UV_decay(self.room.lower_zone.UV_fluence, wavelength = self.room.lower_zone.UV_wavelength)

                self._total_infectivity['lower'] *= (1 - _lower_UV_decay_rate * self.time_step_size)

                self._total_infectivity['lower'] *= (1 - _lower_volumetric_decay_rate * self.time_step_size )

                #if room is zonal, add interzonal rates and calculate total decay rate for upper zone
                if self.room.upper_zone is not None: 
                    assert self.room.inter_zonal_rate is not None 
                    assert self.room.net_flow_rate is not None
                    _upper_volumetric_decay_rate = (self.room.upper_zone.external_air_outflow_rate  * self.room.volume \
                                                + self.room.net_flow_rate * self.room.net_flow_outflow_fraction * self.room.volume \
                                                + self.room.upper_zone.filtration_air_change_rate_roomvol_hourly * self.room.volume \
                                                + self._cache_pathogen_deactivation_rate) / 3600 \
                                                / self.room.upper_zone.volume
                    
                    self.checksmallfactor(_upper_volumetric_decay_rate * self.time_step_size)
                    
                    self._total_infectivity['upper'] *= (1 - _upper_volumetric_decay_rate * self.time_step_size) 

                    _upper_UV_decay_rate = self.pathogen.UV_decay(self.room.upper_zone.UV_fluence, wavelength=self.room.upper_zone.UV_wavelength)

                    self._total_infectivity['upper'] *= (1 - _upper_UV_decay_rate * self.time_step_size)

                    #coupling between the zones
                    if self.room.net_flow_rate > 0:  

                        _lower_to_upper_rate = (self.room.inter_zonal_rate + self.room.net_flow_rate) / 3600 \
                                            * self.room.volume / self.room.lower_zone.volume                    
                        _upper_to_lower_rate = (self.room.inter_zonal_rate) / 3600 \
                                            * self.room.volume / self.room.upper_zone.volume
                        
                    else:
                        _lower_to_upper_rate = (self.room.inter_zonal_rate) / 3600 \
                                            * self.room.volume / self.room.lower_zone.volume
                        _upper_to_lower_rate = (self.room.inter_zonal_rate - self.room.net_flow_rate) / 3600 \
                                            * self.room.volume / self.room.upper_zone.volume
                        
                    self.checksmallfactor(_lower_to_upper_rate * self.time_step_size)
                    self.checksmallfactor(_upper_to_lower_rate * self.time_step_size)
                    
                    _lower_to_upper = self._total_infectivity['lower'] * _lower_to_upper_rate * self.time_step_size 
                    _upper_to_lower = self._total_infectivity['upper'] * _upper_to_lower_rate * self.time_step_size

                    self._total_infectivity['upper'] += _lower_to_upper - _upper_to_lower
                    self._total_infectivity['lower'] += _upper_to_lower - _lower_to_upper

                #setttling decay
                _settling_decay = stokes_velocity(self.droplet_radii, self.room, self.pathogen, critical_radius=self.critical_radius)/self.room.height
                self._total_infectivity['lower'] *= ( 1 - _settling_decay * self.time_step_size)
                self._total_infectivity['upper'] *= ( 1 - _settling_decay * self.time_step_size)

                #update running data
                self.append_data([self.count_susceptible(), 
                                self.count_infected(),
                                self.count_infected_not_quarantined(),
                                self.count_dead(),
                                self.count_quarantined(), 
                                self.count_recovered(),
                                self._total_infectivity['lower'].sum(),
                                self._total_infectivity['upper'].sum(),
                                self.time
                                ])

                #update time and progress bar
                self.time += self.time_step_size
                if task_id is not None and progress is not None: progress.update(
                    task_id, 
                    advance=self.time_step_size, 
                    num_infected=self.count_infected(), 
                    viral_load='{:.2E}'.format(int(self._total_infectivity['lower'].sum())), 
                    time=int(self.time))

        if self.model_break_continue_times is not None: 
            break_times = self.model_break_continue_times[::2]
            continue_times = self.model_break_continue_times[1::2]

            for break_step in range(len(break_times)):
                simulate_step(end_time=break_times[break_step], occupant_freeze=False)
                simulate_step(end_time=continue_times[break_step], occupant_freeze=True)

        simulate_step(end_time=self.max_time, occupant_freeze=False)
        
        self.output_data = pd.DataFrame(columns=self.data_headers, data=self.data)

        return self.output_data

    def occupant_update(self, _occupant: humans.Occupant):
        """Update occupant state based on current state and infection probability"""
        if _occupant.details != None:
            if convert_to_seconds(_occupant.details['quarantine_time']['unit'],_occupant.details['quarantine_time']['value']) < self.time:
                _occupant.quarantine()

        if _occupant.susceptible:
            prob_infection = np.sum(_occupant.incoming_factors() * self._total_infectivity['lower'], dtype=np.double) * self.time_step_size / self.room.lower_zone.volume
            if random.random() < prob_infection: _occupant.infect(time=self.time, pathogen=self.pathogen)
        if _occupant.infected:
            prob_death = _occupant.pathogen.deathFunc(self.time - _occupant.time_infected) * self.time_step_size
            if random.random() < prob_death: 
                _occupant.kill()

            elif (self.time - _occupant.time_infected) > _occupant.pathogen.time_to_symptoms and _occupant.is_symptomatic and _occupant.comply_quarantine:
                _occupant.quarantine()

            elif (self.time - _occupant.time_infected) > _occupant.pathogen.time_to_recovery:
                _occupant.recover()

        return _occupant
    
    def append_data(self, new_data: list):
        """Append data from timestep to self.data"""
        self.data.append(new_data)

"""
Utility functions for modeling
"""
@jit(nopython=True)
def calculate_droplet_volume(droplet_radius: np.ndarray) -> np.ndarray:
    """Calculates the volume of a droplet given its radius (duh)"""
    droplet_volume = 4/3 * np.pi * droplet_radius**3
    return droplet_volume

def calculate_critical_radius(pathogen: pathogens.Pathogen,room: environments.Room,air_density = 1.293) -> float:
    """Critical radius of a droplet, below which it stays suspended in the air"""

    return np.sqrt((9 * room.external_air_change_rate * room.height * room.air_viscosity)/(2 * (pathogen.droplet_density - air_density) * const.g))

def stokes_velocity(radius: npt.NDArray[np.double], room: environments.Room, pathogen: pathogens.Pathogen, air_density = 1.293, critical_radius: float = 0) -> np.ndarray:
    """Calculates the stokes velocity of a droplet, and apply to droplets bigger than critical radius:
    radius: droplet radius
    room: Room object (for height and air viscosity)
    pathogen: Pathogen object (for dropet density)
    air_density: air density
    critical_radius: critical radius of droplet, below which it stays suspended in the air"""
    return np.piecewise(radius, [radius < critical_radius, radius >= critical_radius], [lambda r: 0, lambda r: 2 * const.g * r**2 * (pathogen.droplet_density - air_density)/(9 * room.air_viscosity)])

def Supermodel(
        input: list[tuple[Model, int]], 
        max_threads=4, 
        to_csv: bool = False, 
        output_path = setup_params['environment']['output_path'], 
        other_info: str = '', 
        multithread=False,
    ) -> list[Future]:
    """Main function for running multiple models in parallel. Runs the model concurrently with the specified number of repeats, returns list of the output dataframes, for the user to do as they please with.
    model: Model object to run
    repeats: number of times to run the model
    max_threads: number of threads to run concurrently
    to_csv: if True, output data to csv
        output_path: path to output csvs to, defaults to setup.toml value
        other_info: string to append to output csv name
    multithread: if True, use multithreading, else execute sequentially. Currently havily IO limited and needs to be refactored to use ProcessPoolExecutor, but Model not picklable (yet)"""

    #use Rich to make nice progress bars
    progress = Progress(
        *Progress.get_default_columns(),
        TextColumn("[red]# infected: [bold]{task.fields[num_infected]}"),
        TextColumn("[purple]viral load: [bold]{task.fields[viral_load]}"),
        TextColumn("[cyan]model time: [bold]{task.fields[time]}"),
    )

    results = []

    if multithread:
        with progress:
            with ThreadPoolExecutor(max_workers=max_threads) as pool:
                for model, repeats in input:
                    for repeat_no in range(repeats):
                        _model: Model = copy.deepcopy(model)
                        task_id = progress.add_task("{}: repeat {} of {}".format(_model.name,repeat_no+1, repeats),start=False, num_infected=0, viral_load=0, time=0)
                        _future = pool.submit(run_model,_model, task_id, progress)
                        results.append(_future)
                        if to_csv: _future.result().to_csv(output_path + _model.name + '_rep_' + str(repeat_no) + other_info +'.csv')
    
    else:
        with progress:
            for model, repeats in input:
                for repeat_no in range(repeats):
                    _model: Model = copy.deepcopy(model)
                    _future = run_model(_model, None, None)
                    results.append(_future)
                    if to_csv: _future.to_csv(output_path + _model.name + '_rep_' + str(repeat_no) + other_info +'.csv')

    return results

def run_model(model: Model, task_id: Optional[TaskID],  progress:Optional[Progress]) -> pd.DataFrame:
    if task_id is not None and progress is not None: progress.start_task(task_id)
    _model = copy.deepcopy(model)
    for _occupant in _model.occupants:
        _occupant.recalc_random()
    if task_id is not None and progress is not None: progress.update(task_id, total=_model.max_time)

    _model.run(progress, task_id)

    return _model.output_data

def Supermodel_experimental(input: list[tuple[Model, int]], max_threads=4, to_csv: bool = False, output_path = setup_params['environment']['output_path'], other_info: str = '') -> list[Future]:
    """experimental version of Supermodel - do not use"""

    #use Rich to make nice progress bars
    progress = Progress(
        *Progress.get_default_columns(),
        TextColumn("[red]# infected: [bold]{task.fields[num_infected]}"),
        TextColumn("[purple]viral load: [bold]{task.fields[viral_load]}"),
        TextColumn("[cyan]model time: [bold]{task.fields[time]}"),
    )

    results = []

    with progress:
        with ProcessPoolExecutor(max_workers=max_threads) as pool:
            for model, repeats in input:
                for repeat_no in range(repeats):
                    _model: Model = copy.deepcopy(model)
                    task_id = progress.add_task("{}: repeat {} of {}".format(_model.name,repeat_no+1, repeats),start=False, num_infected=0, viral_load=0, time=0) if _model.verbose else None
                    _future = pool.submit(run_model,_model, task_id, progress)
                    results.append(_future)
                    if to_csv: _future.result().to_csv(output_path + _model.name + '_rep_' + str(repeat_no) + other_info +'.csv')

    return results

