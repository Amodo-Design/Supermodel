#Tools for using the inbuilt spreadsheet case studies

if __name__ == '__main__':
    raise EnvironmentError('This file is for importing, not running directly')

from . import distributions, environments, humans, interventions, modeling, pathogens, schema
from .classes import convert_to_seconds
import numpy as np
import tomli
from rich import print
import numpy.typing as npt
from supermodel import CONFIG_PATH

if __name__ == '__main__':
    raise EnvironmentError('This file is for importing, not running directly')

#Importing data folder from setup
if CONFIG_PATH is not None:
    with open(CONFIG_PATH, 'rb') as setup_file:
        setup_params = tomli.load(setup_file)
else:
    raise Exception('CONFIG_PATH not set, please run supermodel.loadconfig(path)')

pathogen_path = setup_params["pathogen"]['pathogen_path']
case_study_path = setup_params['case_studies']['case_study_path']



#needs changing for new TOML format
def distribution_parser(droplet_distributions:dict[str, float]) -> list[distributions.Distribution]:
    "Gives a distribution curve from string in data"

    distribution_list= []

    for distribution_name, relative_value in droplet_distributions.items():
        
        match distribution_name:
            case "singing":
                distribution_list.append( humans.generate_loud_singing_dist(weight=relative_value))
            case "heavy_breathing":
                distribution_list.append( humans.generate_heavy_breathing_dist(weight=relative_value))
            case "breathing":
                distribution_list.append( humans.generate_breathing_dist(weight=relative_value))
            case "intermittent_shouting":
                distribution_list.append( humans.generate_intermittent_shouting_dist(weight=relative_value))
            case "speaking_normal":
                distribution_list.append( humans.generate_intermediate_speech_dist(weight=relative_value))
            case "speaking_quiet breathing half":
                distribution_list.append( humans.generate_quiet_speech_half_dist(weight=relative_value))
            case "speaking_quiet":
                distribution_list.append( humans.generate_quiet_speech_dist(weight=relative_value))
            case _:
                raise ValueError("Distribution name not recognized: {}".format(distribution_name))

    return distribution_list

def PPE_parser(_ppe_text_list: list[str]) -> tuple[list[interventions.PPE], list[tuple[str, int, interventions.PPE]]]:
    '''Parse ppe list from spreadsheet'''
    ppe_list = []
    ppe_tuples = []

    for PPE_item in _ppe_text_list:
        PPE_param_list = PPE_item.split("_")
        target_type = PPE_param_list[0]
        if target_type not in ['S', 'I']:
            raise ValueError("PPE target type not recognized: {}".format(target_type))
        target_number = int(PPE_param_list[1])
        PPE_type = PPE_param_list[2]

        match PPE_type:
            case "SM":
                # Surgical Mask
                resp_pen_in_curve = interventions.generate_surgical_mask_in_dist()
                resp_pen_out_curve = interventions.generate_surgical_mask_out_dist()
             
            case "CM":
                # Cloth Mask
                resp_pen_in_curve = interventions.generate_cloth_mask_in_dist()
                resp_pen_out_curve = interventions.generate_cloth_mask_out_dist()

            case "N95":
                # N95
                resp_pen_in_curve = interventions.generate_n95_mask_in_dist()
                resp_pen_out_curve = interventions.generate_n95_mask_out_dist()

            case "PAPR":
                # PAPR
                resp_pen_in_curve = distributions.generate_HEPA_filter_dist()
                resp_pen_out_curve = distributions.generate_zero_dist()
            
            case "N95E":
                # N95 Elastomeric
                resp_pen_in_curve = interventions.generate_N95_elasromeric_nofit_in_dist()
                resp_pen_out_curve = distributions.generate_zero_dist()

            case "N95E-SC":
                # N95 Elastomeric with source control
                resp_pen_in_curve = interventions.generate_N95_elasromeric_nofit_in_dist()
                resp_pen_out_curve = interventions.generate_N95_elasromeric_nofit_in_dist()

            case "N95E-SC-FIT":
                # N95 Elastomeric with source control and fit
                resp_pen_in_curve = interventions.generate_N95_elastomeric_in_dist()
                resp_pen_out_curve = interventions.generate_N95_elastomeric_in_dist()

            case "N95E-FIT":
                # N95 Elastomeric with fit
                resp_pen_in_curve = interventions.generate_N95_elastomeric_in_dist()
                resp_pen_out_curve = distributions.generate_zero_dist()

            case _:
                raise ValueError("PPE type not recognized: {}".format(PPE_type))
            
        PPE_Instance = interventions.PPE(
            resp_pen_in_curve=resp_pen_in_curve,
            resp_pen_out_curve=resp_pen_out_curve,
            )

        ppe_tuples.append((target_type, target_number, PPE_Instance))
        
        ppe_list.append(PPE_Instance)

    return (ppe_list, ppe_tuples)
    
def load_case_study_file(path:str = case_study_path) -> dict:
    """Loads the case study sheet. Takes path to relevant TOML and returns dict that conforms with the correct case-study data-structure."""

    print("[cyan]Loading case study file at [italic]" + path)
    with open(path, 'rb') as file:
        case_study_raw = tomli.load(file)

    names = [index for index, value in case_study_raw.items()]

    if len(set(names)) != len(names):
        raise ValueError("Duplicate case study names found in case study file")

    return case_study_raw

def load_pathogen_file(path:str = pathogen_path) -> dict:
    """Loads the case study sheet. Takes path and returns dataframe."""

    print("[cyan]Loading pathogen file at [italic]" + path)
    with open(path, 'rb') as file:
        pathogen_raw = tomli.load(file)

    names = [name for name, data in pathogen_raw.items()]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate pathogen names found in pathogen sheet")

    return pathogen_raw

def pathogen_from_dataframe(data_frame: dict, pathogen_input: str):
    #extract pathogen data from a Pandas dataframe and return a Pathogen object


    if not isinstance(pathogen_input, str):
        raise Exception("Pathogen name must be a string")
    
    for data in data_frame:
        pathogen_name = data['name']
        # deal with populations
        if "populations" in data:
            populations = data["populations"]
        else:
            populations= None


        if pathogen_input == pathogen_name:
            pathogen = pathogens.Pathogen(

                name = pathogen_name,
                cp = data["cv"], # Virions per m3 fluid
                pi = data["pi"],  # (infection chance/virion)
                pathogen_size = data["pathogen_size"], # (m) - diameter
                liquid_deactivation_curve = pathogens.generate_covid_19_liq_deactivation_curve(),
                asymptomatic_proportion = data["asymptomatic_proportion"], # proportion of infections that are asymptomatic
                lethality = data["lethality"],
                time_to_recovery = data["time_to_recovery"],# Time to recovery in seconds
                time_to_death = data["time_to_death"], # Time to death in seconds
                time_to_symptoms = data["time_to_symptoms"],
                infectivity_curve = distributions.Distribution_Lognormal(
                    mean = data["inf_curve_mean"],
                    std_dev = data["inf_curve_std_dev"],
                    y_scale = data["inf_curve_scale"],
                    y_offset = data["inf_curve_offset"],
                    x_offset=0,
                    x_scale = 1/(24*3600) # Translate time from seconds to days
                ),
                UV_deactivation_rates=data["UV_deactivation"], # mJ/cm^2
                populations=populations,
            )
            return pathogen
    
    raise Exception("No pathogen with name {} found in spreadsheet".format(pathogen_input))

def construct_case_study(
        requested_name: str,
        case_study_dict: dict = load_case_study_file(),
        pathogen_dict: dict = load_pathogen_file(),
        ) -> modeling.Model:
    """Parses a dataframe of case studies into a list of case study models, ready to be run, and searches for the name of the requested case studies.

    requested name: list of case study names to be parsed. Default is all case studies.
    case_study_df: dataframe of case studies. Default is the case study sheet in data/
    pathogen_df: dataframe of pathogens. Default is the pathogen sheet in data/
    
    returns: list of case study Model objects"""

    if not requested_name:
        raise Exception("No case studies requested")

    for data in case_study_dict['case_study']:

        case_study_name = str(data['name']).strip()
        if case_study_name != requested_name:
            continue

        #construct zone_data: 
        if 'zone_data' in data:
            UV_type = "upper_room"
            if "UV" in data: print("Assuming upper room UV given zonal model, ignoring UV.xx in setup.toml")
            environment_type = 'zonal'
            iszonal = True
            room_height = sum([zone_data['height'] for zone_name, zone_data in data['zone_data']['zones'].items()])
            room_filter_air_change_rate = sum([zone_data['filtration_air_change_rate'] for zone_name, zone_data in data['zone_data']['zones'].items()])
            room_external_air_change_rate = sum([zone_data['external_air_change_rate'] for zone_name, zone_data in data['zone_data']['zones'].items()])
            zone_list = []
            interchange_rate = data['zone_data']['beta']
            net_flow_rate = data['zone_data']['net_flow']
            net_flow_outflow_fraction = data['zone_data']['net_flow_outflow_fraction']
            for zone_name, zone_data in data['zone_data']['zones'].items():
                if 'UV' in zone_data: zone_list.append(environments.zone(
                    height = zone_data['height'],
                    external_air_change_rate = zone_data['external_air_change_rate'],
                    filtration_air_change_rate = zone_data['filtration_air_change_rate'],
                    UV_fluence=zone_data['UV']['fluence'],
                    UV_wavelength=zone_data['UV']['wavelength'],
                ))
                else: zone_list.append(environments.zone(
                    height = zone_data['height'],
                    external_air_change_rate = zone_data['external_air_change_rate'],
                    filtration_air_change_rate = zone_data['filtration_air_change_rate'],
                ))

        else: 
            environment_type = 'non-zonal'
            interchange_rate = None
            net_flow_rate = None
            net_flow_outflow_fraction = None
            iszonal = False
            if "UV" in data: UV_type = data["UV"]["type"]
            else: UV_type = None
            room_height = data["H"]
            room_fiter_air_change_rate = data["fach"]
            room_external_air_change_rate = data["ach"]
            if 'UV' in data:
                room_UV_fluence = data["UV"]["fluence"]
                room_UV_wavelength = data["UV"]["wavelength"]
            else:
                room_UV_fluence = 0
                room_UV_wavelength = 0
            zone_list = [
                environments.zone(
                    external_air_change_rate=room_external_air_change_rate,
                    height=room_height,
                    filtration_air_change_rate=room_fiter_air_change_rate,
                    UV_fluence=room_UV_fluence,
                    UV_wavelength=room_UV_wavelength,
                )
            ]
        #read in data from spreadsheet to pass to model and occupant objects

        pathogen_name = str(data["pathogen"]["name"]).strip()
        temperature_C = data['avg_temp_c']
        temperature_K = temperature_C + 273.15
        num_occupants = data["occupants"] 
        num_initial_infected = data["i0"] 
        room_area_average = data["A"] 
        initial_infection_time = data["inf_t0"]*24*3600 


        breathing_flowrate = data["Qb"]/3600
        quarantine_compliance_average = data["avg_quarantine_compliance"]
        ppe_compliance_average = data["avg_ppe_compliance"]
        quarantine_bool = data["quarantine"]

        if 'data' in data: 
            data_infections = np.array(data["data"]["infections"])
            data_times = np.array([convert_to_seconds(data["data"]["times"]['unit'], value) for value in data["data"]["times"]['values']])
        else:
            data_infections = None
            data_times = None
        #this  is wring given new TOML format, fix this

        pathogen = pathogen_from_dataframe(pathogen_dict['pathogen'], pathogen_name)
        droplet_distributions = distribution_parser(data['droplet_distribution'])

        #droplet radii from setup.toml

        num_droplet_radii = setup_params['modeling']['num_droplet_radii']
        min_droplet_radius = setup_params['modeling']['min_droplet_radius']
        max_droplet_radius = setup_params['modeling']['max_droplet_radius']
        droplet_radius_step = (max_droplet_radius - min_droplet_radius)/num_droplet_radii
        droplet_radii = np.arange(min_droplet_radius, max_droplet_radius, droplet_radius_step)


        #extract max time depeding on unit
        max_time = convert_to_seconds(data["max_time"]['unit'],  data["max_time"]['value'])

        #extract data times
        
        #extract PPE types
        if 'PPE' in data: ppe_list, ppe_tuples = PPE_parser(str(data["PPE"]).split(","))
        else: ppe_list, ppe_tuples = None, None

        #extract filter types
        if 'filter distribution' in data: 
            match str(data["filter_distribution"]):
                case "perfect":
                    filter_distribution = distributions.generate_perfect_filter_dist()
                
                case "HEPA":
                    filter_distribution = distributions.generate_HEPA_filter_dist()

                case 'onemicron':
                    filter_distribution = distributions.generate_one_micron_filter_dist()

                case _:
                    raise ValueError("Filter type not recognized: {}".format(str(data["filter_distribution"])))
        else: filter_distribution = None

        #extract model break times
        if 'model_break_times' in data: 
            model_break_times = np.array([convert_to_seconds(
                unit = data['model_break_times']['unit'], 
                value = value) for value in data['model_break_times']['values']])
        
        else: model_break_times = None

        #construct maximum time for model
        _max_time = 0
        if model_break_times is not None:
            _max_time = max(_max_time, model_break_times.max())
        if data_times is not None:
            _max_time = max(_max_time, data_times.max())
        
        if max_time < _max_time:
            raise ValueError("max_time must be greater than the last data time and model last model break time")
        


        #construct environment, occupants, and model
        initial_occupant_list = [
            humans.Occupant(
                activity_distributions=droplet_distributions,
                pathogen = pathogen,
                quarantine_compliance=quarantine_compliance_average,
                ppe_compliance=ppe_compliance_average,
                respiratory_ppe=ppe_list,
                droplet_radii=droplet_radii,
                breathing_flowrate=breathing_flowrate,
                ppe_fit=1,
                ) for _ in range(num_occupants)]

        room = environments.Room(
            environment_type=environment_type,
            zones=zone_list,
            area=room_area_average,
            temperature=temperature_K,
            filter_distribution=filter_distribution,
            inter_zonal_rate=interchange_rate,
            net_flow_rate=net_flow_rate,
            net_flow_outflow_fraction=net_flow_outflow_fraction,
        )

        model = modeling.Model(
            UV_type=UV_type,
            iszonal=iszonal,
            room=room,
            occupants=initial_occupant_list,
            pathogen=pathogen,
            model_break_continue_times=model_break_times,
            max_time=max_time,
            quarantine=quarantine_bool,
            time_data=data_times,
            infection_data=data_infections,
            name=case_study_name,
            droplet_radii=droplet_radii,
            droplet_radius_step=droplet_radius_step,
        )

        for occupant in model.occupants:
            occupant.totals = model.totals

        for i in range(num_initial_infected):
            model.occupants[i].infect(time=initial_infection_time, pathogen=pathogen)
        
        # specific occupant data
        if 'unique_occupants' in data:
            for name, unique_data in data['unique_occupants'].items():
                index = unique_data['index']
                if index >= num_occupants:
                    raise ValueError("Unique occupant index out of range")
                model.occupants[index].details = unique_data

        
        return model
    
    raise Exception("No case study with name {} found in spreadsheet".format(requested_name))


