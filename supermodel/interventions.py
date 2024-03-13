# Classes and functions relating to interventions

if __name__ == "__main__":
    raise EnvironmentError("This file is not meant to ran as main. Please import it into another file.")

from . import distributions
import numpy as np

class PPE:
    def __init__(
        self,
        resp_pen_in_curve: distributions.Distribution,
        resp_pen_out_curve: distributions.Distribution,
        ):
        self.resp_pen_in_curve = resp_pen_in_curve
        self.resp_pen_out_curve = resp_pen_out_curve

"""
Generate intervention related distributions, data from ../research/mask_curves.xlsx
"""
#N95 masks


def generate_n95_mask_out_dist():
    return distributions.Distribution_Interpolated(
        xs = np.array([0.02,0.04,0.06,0.08,0.1,0.175,0.25,0.3,0.5,0.7,1,2,3,4,5]),
        ys = np.array([0,0,0.002,0.005,0.007,0.01,0.02,0.03,0.03,0.03,0.03,0.02,0.01,0.01,0]),
        input_scale = 1E-6
    )


def generate_n95_mask_in_dist():
    return distributions.Distribution_Interpolated(
        xs = np.array([0.02,0.04,0.06,0.08,0.1,0.175,0.25,0.3,0.5,0.7,1,2,3,4,5]),
        ys = np.array([0,0.002,0.005,0.007,0.01,0.02,0.04,0.07,0.07,0.065,0.06,0.04,0.03,0.02,0]),
        input_scale = 1E-6
    )

#surgical masks


def generate_surgical_mask_out_dist():
    mask_distribution = distributions.Distribution_Interpolated(
        xs=np.array([0.05,0.125,0.25,0.35,0.5,1,2.5,3,3.5]),
        ys=np.array([0.8,0.75,0.6,0.35,0.3,0.25,0.1,0,0]),
        input_scale = 1E-6
    )
    return mask_distribution


def generate_surgical_mask_in_dist():
    return distributions.Distribution_Interpolated(
        xs=np.array([0.05,0.125,0.25,0.35,0.5,1,2.5,3,3.5]),
        ys=np.array([0.75,0.72,0.71,0.7,0.65,0.55,0.1,0.1,0]),
        input_scale = 1E-6
    )

#cloth masks


def generate_cloth_mask_in_dist():
    return distributions.Distribution_Interpolated(
        xs = np.array([0.05, 0.125, 0.25, 0.35, 0.5, 1, 2.5, 3, 3.5]),
        ys = np.array([0.9, 0.8, 0.85, 0.9, 0.85, 0.55, 0.25, 0.1, 0]),
        input_scale = 1E-6
    )


def generate_cloth_mask_out_dist():
    return distributions.Distribution_Interpolated(
        xs= np.array([0.05, 0.125, 0.25, 0.35, 0.5, 1, 2.5, 3, 3.5]),
        ys= np.array([1, 1, 1, 0.75, 0.55, 0.45, 0.15, 0.1, 0]),
        input_scale = 1E-6
    )

#elastomeric respirators

def generate_P100_elastomeric_in_dist():
    return distributions.Distribution_Interpolated(
        xs= np.array([0.011170, 0.016206, 0.025278, 0.046472, 0.081600, 0.109096, 0.143165, 0.189605, 0.265472, 0.388988]),
        ys= np.array([0.00010833333333333300, 0.00011904761904761900, 0.00014008620689655200, 0.00015625000000000000, 0.00019005847953216400, 0.00022108843537415000, 0.00029279279279279300, 0.00042207792207792300, 0.00045138888888888900, 0.00049242424242424300]),
        input_scale=1e-6)

def generate_N95_elastomeric_in_dist():
    return distributions.Distribution_Interpolated(
        xs= np.array([0.0108606681844327, 0.0134507725799335, 0.017417720933784, 0.0275710972359734, 0.0401661963292717, 0.0553455518696337, 0.0941113533727602, 0.134461388021049, 0.218433339329763, 0.385247687107938, 0.85,1.50,2.50,4.00,7.50]),
        ys= np.array([0.00101824817518248, 0.00161271676300578, 0.00255963302752294, 0.00442857142857143, 0.00774999999999999, 0.00871875, 0.00774999999999999, 0.00634090909090908, 0.00619999999999999, 0.00536538461538461, 0.005841999482890,0.003105371512749,0.001000000000000,0.000466412000562,0.000351351105962]),
        input_scale=1e-6)

def generate_N95_elasromeric_nofit_in_dist():
    return distributions.Distribution_Interpolated(
        xs = np.array([0.0108606681844327, 0.0134507725799335, 0.017417720933784, 0.0275710972359734, 0.0401661963292717, 0.0553455518696337, 0.0941113533727602, 0.134461388021049, 0.218433339329763, 0.385247687107938, 0.85,1.50,2.50,4.00,7.50]),
        ys = np.array([0.0183192008582428, 0.0290142256367545, 0.0460501012399866, 0.0796739846850563, 0.139429473198848, 0.156858157348705, 0.139429473198848, 0.114078659889967, 0.111543578559079, 0.096528096829972, 0.105102827139013, 0.0558684276269987, 0.0179908997675933, 0.00839117155251952, 0.00632112253058883]),
        input_scale=1e-6)