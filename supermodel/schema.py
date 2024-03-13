# This file contains the schema for the TOML files that are used to configure the model.


from schema import Schema, And, Or, Optional
import tomli
from supermodel import CONFIG_PATH

number = Or(int, float)

def input_validation(environment_setup_path:str):

    setup_schema = Schema({
        "case_studies": {
            "case_study_path": And(str, len),
        },
        "pathogen": {
            "pathogen_path": And(str, len),
        },
        "modeling": {
            "num_droplet_radii": And(int, lambda n: n > 0),
            "min_droplet_radius": And(number, lambda n: n >= 0),
            "max_droplet_radius": And(number, lambda n: n > 0),
        },
        "environment": {
            "output_path": And(str, len),
            "graphics_path": And(str, len),
        }
    })

    if CONFIG_PATH is not None:
        with open(CONFIG_PATH, 'rb') as setup_file:
            setup_params = tomli.load(setup_file)

        print(f'Checking setup TOML file at {environment_setup_path}')
        setup_schema.validate(setup_params)
    else:
        raise Exception('CONFIG_PATH not set, please run supermodel.loadconfig(path)')

    case_study_path = setup_params['case_studies']['case_study_path']
    pathogen_path = setup_params['pathogen']['pathogen_path']

    tomlparser(case_study_path, pathogen_path)


def tomlparser(case_study_path: str, pathogen_path: str):

    with open(case_study_path, mode='rb') as tomlfile:
        indict = tomli.load(tomlfile)

        print(f'Parsing TOML file at {case_study_path}')

        for data in indict["case_study"]:

            case_study_schema.validate(data)
            print(f"Parsing {data['name']} case study")

    with open(pathogen_path, mode='rb') as tomlfile:
        indict = tomli.load(tomlfile)

        print(f'Parsing TOML file at {pathogen_path}')

        for data in indict['pathogen']:

            pathogen_schema.validate(data)
            print(f"Parsed {data['name']} pathogen")



time_units = ["days", "hours", "minutes", "seconds"]
        
pathogen_schema = Schema({
        Optional("comments"): And(str, len),
        "name": And(str, len),
        "cv": And(int, lambda n: n > 0),
        "pi": And(int, lambda n: n >= 0),
        "lambda_v": And(float, lambda n: n >= 0),
        "pathogen_size": And(float, lambda n: n >= 0),
        "lethality": And(float, lambda n: 0 <= n <= 1),
        "asymptomatic_proportion": And(float, lambda n: 0 <= n <= 1),
        "time_to_recovery": And(int, lambda n: n > 0),
        "time_to_death": And(int, lambda n: n > 0),
        "inf_curve_mean": And(int, lambda n: n >= 0),
        "inf_curve_std_dev": And(int, lambda n: n >= 0),
        "inf_curve_scale": And(float, lambda n: n >= 0),
        "inf_curve_offset": And(int, lambda n: n >= 0),
        
        
        
        
        
        "time_to_symptoms": And(int, lambda n: n > 0),
        Optional("UV_deactivation"):{
            Optional("222"): And(float, lambda n: n >= 0),
        },
        Optional("populations"): {
            str: {
                Optional("222"): And(float, lambda n: n >= 0),
                "population_fraction": And(float, lambda n: 0 <= n <= 1),
            },
        }
})

case_study_schema = Schema({
    "name": And(str, len),
    "occupants": And(int, lambda n: n > 0),
    "i0": And(int, lambda n: n >= 0),
    "ach": And(number, lambda n: n >= 0),
    "pathogen": {
        "name": And(str, len),
        Optional("pathogen_pops"): And(str, len),
    },
    "A": And(number, lambda n: n > 0),
    "H": And(number, lambda n: n > 0),
    "fach": And(number, lambda n: n >= 0),
    Optional("filter_distribution"): And(str, len),
    "inf_t0": And(number),
    "max_time": {
        "unit": And(str, lambda n: n in time_units),
        "value": And(number, lambda n: n > 0)
    },
    "Qb": And(number, lambda n: n >= 0),
    Optional("data") : {
        "times" : {
            "unit": And(str, lambda n: n in time_units),
            "values": [And(number, lambda n: n >= 0)]
        },
        "infections": [And(number, lambda n: n >= 0)]
    },
    "avg_temp_c": And(number),
    "droplet_distribution": {
        Optional("singing"): And(number, lambda n: n >= 0),
        Optional("heavy_breathing"): And(number, lambda n: n >= 0),
        Optional("breathing"): And(number, lambda n: n >= 0),
        Optional("intermittent_shouting"): And(number, lambda n: n >= 0),
        Optional("speaking_normal"): And(number, lambda n: n >= 0),
        Optional("speaking_quiet"): And(number, lambda n: n >= 0),
    },
    Optional("PPE"): And(str, len),
    Optional("model_break_times"): {
        "unit": And(str, lambda n: n in time_units),
        "values": [And(number, lambda n: n >= 0)]
    },
    "avg_quarantine_compliance": And(number, lambda n: 0 <= n <= 1),
    "avg_ppe_compliance": And(number, lambda n: 0 <= n <= 1),
    "quarantine": bool,
    Optional("infectivity"): And(number, lambda n: n >= 0),
    Optional("susceptibility"): And(number, lambda n: n >= 0),
    Optional("beta"): And(number, lambda n: n >= 0),
    Optional("UV"): {
        "fluence": And(number, lambda n: n >= 0),
        "wavelength": And(number, lambda n: n >= 200),
        "type"  : And(str, lambda n: n in ["upper_room", "whole_room"]),
    },
    Optional("zone_data"): {    
        "beta": And(number, lambda n: n >= 0),
        "net_flow": number,
        "net_flow_outflow_fraction": And(number, lambda n: 0 <= n <= 1),
        "zones":{
            "lower_zone": {
                "height": And(number, lambda n: n > 0),
                "external_air_change_rate": And(number, lambda n: n >= 0),
                Optional("filtration_air_change_rate"): And(number, lambda n: n >= 0),
                Optional("UV"): {
                    "fluence": And(number, lambda n: n >= 0),
                    "wavelength": And(number, lambda n: n >= 200),
                },
            },
            "upper_zone": {
                "height": And(number, lambda n: n > 0),
                "external_air_change_rate": And(number, lambda n: n >= 0),
                Optional("filtration_air_change_rate"): And(number, lambda n: n >= 0),
                Optional("UV"): {
                    "fluence": And(number, lambda n: n >= 0),
                    "wavelength": And(number, lambda n: n >= 200),
                }
            }
        }
    }
})