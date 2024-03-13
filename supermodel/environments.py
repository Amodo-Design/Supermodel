
"""
Classes, relating to the model environment
"""
from __future__ import annotations
from optparse import Option
from pickle import FLOAT
if __name__ == "__main__": 
    raise Exception("This file is not meant to run by itself, please import it")
import numpy as np
from typing import Optional
import numpy.typing as npt

class zone:
    """Stores data about each zone in in the model"""

    def __init__(
            self,
            height: float,
            external_air_change_rate: float,
            filtration_air_change_rate: float,
            UV_fluence: float = 0,
            UV_wavelength: Optional[float] = None,
            ):
        self.height = height
        self.external_air_outflow_rate = external_air_change_rate
        self.filtration_air_change_rate_roomvol_hourly = filtration_air_change_rate
        self.UV_fluence = UV_fluence
        self.UV_wavelength = UV_wavelength
        self._volume = 0

    @property
    def volume(self) -> float:
        if self._volume == 0:
            raise ValueError("Zone volume has not been set, or zone height is zero")
        return self._volume
    @volume.setter
    def volume(self, volume: float):
        self._volume = volume

class Room:
    """Stores the data for a room in the model"""

    def __init__(
        self,                 
        area, # m2
        zones: list[zone],
        environment_type, # zonal or non-zonal
        inter_zonal_rate: Optional[float] = None, # room vol/hr
        net_flow_rate: Optional[float] = None, # room vol/hr
        net_flow_outflow_fraction: Optional[float] = None, # constant inflow or outflow
        air_viscosity = 1.86E-5 , # Viscosity of air in room Pa s
        temperature = 293,  # Temp of air in room K
        filter_distribution = None, 
        relative_humidity = 0.65,
        UV_wavelength:float = 0,
        UV_fluence:float = 0,

        ):  
        # logic to enforce paramaters for either zonal or non-zonal environment
        self.inter_zonal_rate = inter_zonal_rate
        if net_flow_rate is None: self.net_flow_rate = 0
        else: self.net_flow_rate = net_flow_rate
        if net_flow_outflow_fraction is None: self.net_flow_outflow_fraction = 0
        else: self.net_flow_outflow_fraction = net_flow_outflow_fraction

        # set object attributes
        self.area = area
        self.air_viscosity = air_viscosity
        self.temperature = temperature
        self.filter_distribution = filter_distribution
        self.relative_humidity = relative_humidity
        self.zones = zones
        self.UV_wavelength = UV_wavelength
        self.UV_fluence = UV_fluence

        # set zone volumes
        self.set_zone_volumes()

        match environment_type:
            case "zonal":
                if len(self.zones) != 2:
                    raise ValueError("zonal environments must have two zones")
                if inter_zonal_rate is None:
                    raise ValueError("inter_zonal_rate must be provided for zonal environments")
                if net_flow_rate is None:
                    raise ValueError("net_flow_rate must be provided for zonal environments")
                self.lower_zone = self.zones[0]
                self.upper_zone = self.zones[1]
                self.check_net_flow()
                
            case "non-zonal":
                if len(self.zones) != 1:
                    raise ValueError("non-zonal environments must have one zone")
                self.lower_zone = self.zones[0]
                self.upper_zone = None
            
            case _:
                raise ValueError("environment_type must be one of {}".format('"zonal", "non-zonal"'))
            
    def check_net_flow(self):
        """Ensure that flows are in the correct direction"""
        if not self.upper_zone:
            raise ValueError("Net flow rate is not applicable to non-zonal environments")
        
        assert self.net_flow_rate is not None
        if self.upper_zone.external_air_outflow_rate < self.net_flow_rate:
            raise ValueError("Net flow too positive, air flows out of the intake in the upper zone")
        
        if self.lower_zone.external_air_outflow_rate < -self.net_flow_rate:
            raise ValueError("Net flow too negative, air flows out of the intake in the lower zone")


    def set_zone_volumes(self):
        for zone in self.zones:
            zone.volume = zone.height * self.area
    
    @property
    def volume(self) -> float:
        return self.area * self.height
    
    @property
    def external_air_change_rate(self) -> float:
        
        if self.upper_zone is None:
            return self.lower_zone.external_air_outflow_rate
        return self.lower_zone.external_air_outflow_rate + self.upper_zone.external_air_outflow_rate

    @property
    def height(self) -> float:
        if self.upper_zone is None:
            return self.lower_zone.height
        return self.lower_zone.height + self.upper_zone.height
