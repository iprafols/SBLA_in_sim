"""Utilities for line fitting"""
import copy
import numpy as np
import iminuit
from astropy.io import fits
import h5py

from sbla_in_sim.line_profile_least_squares import LineProfileLeastSquares

SPECIES_DICTS = {
    'Lya': {
        'f': [.4160],
        'Gamma':[4.690E8],
        'wavelength':[1215.67],
    },
    'Lyb': {
        'f': [.0791],
        'Gamma':[5.570E7],
        'wavelength':[1025.7222],
    },
    'Lyg': {
        'f': [.0290],
        'Gamma':[1.280E6],
        'wavelength':[972.5367],
    },
    'Lyother': { # Lyman series (excluding Lya, Lyb and Lyg)
        'f': [
            2.440e-05, 2.640e-05, 2.850e-05, 3.090e-05, 3.350e-05, 3.650e-05, 3.980e-05,
            4.360e-05, 4.780e-05, 5.260e-05, 5.800e-05, 6.430e-05, 7.140e-05, 7.970e-05,
            8.920e-05, 1.000e-04, 1.140e-04, 1.290e-04, 1.480e-04, 1.700e-04, 1.970e-04,
            2.290e-04, 2.700e-04, 3.210e-04, 3.850e-04, 4.680e-04, 5.770e-04, 7.220e-04,
            9.210e-04, 1.200e-03, 1.600e-03, 2.210e-03, 3.180e-03, 4.810e-03, 7.790e-03,
            1.390e-02,
        ],
        'Gamma': [
            1.220e+02, 1.390e+02, 1.580e+02, 1.810e+02, 2.070e+02, 2.390e+02, 2.760e+02,
            3.200e+02, 3.740e+02, 4.380e+02, 5.160e+02, 6.120e+02, 7.290e+02, 8.740e+02,
            1.060e+03, 1.280e+03, 1.580e+03, 1.950e+03, 2.440e+03, 3.070e+03, 3.920e+03,
            5.070e+03, 6.650e+03, 8.850e+03, 1.200e+04, 1.660e+04, 2.340e+04, 3.390e+04,
            5.060e+04, 7.830e+04, 1.260e+05, 2.140e+05, 3.870e+05, 7.560e+05, 1.640e+06,
            4.120e+06
        ],
        'wavelength': [
            912.3236, 912.3532, 912.3853, 912.4199, 912.4575, 912.4983, 912.5428,
            912.5914, 912.6447, 912.7032, 912.7676, 912.8388, 912.9178, 913.0058,
            913.1041, 913.2145, 913.3391, 913.4802, 913.6411, 913.8256, 914.0385,
            914.2860, 914.5762, 914.9192, 915.3289, 915.8237, 916.4290, 917.1805,
            918.1293, 919.3513, 920.9630, 923.1503, 926.2256, 930.7482, 937.8034,
            949.7429
        ]
    },
    'CIV': {
        'f': [1.900000E-01, 9.520000E-02],
        'Gamma':[2.650000E+08, 2.640000E+08],
        'wavelength':[1548.1870, 1550.7720],
    },
    'MgII': {
        'f': [6.210000E-04, 3.510000E-04],
        'Gamma':[1.350000E+06, 1.520000E+06],
        'wavelength':[1239.925300, 1240.394700],
    },
    'OVI': {
        'f': [1.330000E-01, 6.600000E-02],
        'Gamma':[4.160000E+08, 4.090000E+08],
        'wavelength':[1031.912000, 1037.613000],
    },
    'SiIV': {
        'f': [2.550000E-01],
        'Gamma':[8.630000E+08],
        'wavelength':[1402.770000],
    },
    'NeVIII': {
        'f': [1.020000E-01, 5.020000E-02],
        'Gamma':[5.720000E+08, 5.500000E+08],
        'wavelength':[770.409000, 780.324000],
    },
}

MIN_NHI = 1e5
MAX_NHI = 1e24
START_NHI = 1e12

for species_dict in SPECIES_DICTS.values():
    species_dict["max_N"] = MAX_NHI
    species_dict["min_N"] = MIN_NHI
    species_dict["init_N"] = START_NHI

ORDER_FITS = [
    "Lya", "Lyb", "CIV", "MgII", "OVI", "SiIV", "NeVIII", "Lyg", "Lyother"
]

Z_MIN_FACTOR = 0.9
Z_MAX_FACTOR = 1.1
B_MIN_FACTOR = 0.9
B_MAX_FACTOR = 1.1

def fit_lines(rays_base_name, input_extension, noise, start_z, start_b,
              species_dicts=SPECIES_DICTS, order_fits=ORDER_FITS):
    """
    Fit absorption lines in a given spectrum.
    
    Arguments
    ---------
    rays_base_name: str
    Base name used to name the outputs

    input_extension: str
    Extension of the file containing the spectra. Valid extensions are
    ".h5" or ".txt"

    noise: float
    Noise level in the spectra (-1 for no noise)

    start_z: float
    Initial guess for the redshift. Redshift range in the fit set from
    Z_MIN_FACTOR * start_z to Z_MAX_FACTOR * start_z

    start_b: float
    Initial guess for the impact parameter. Impact parameter range in the
    fit set from B_MIN_FACTOR * start_b to B_MAX_FACTOR * start_b

    species_dicts: dict of dicts - default: SPECIES_DICTS
    Top level keys should be the names of all the species given
    in order_fits. The entries should be dictionaries containing all
    relevant parameters needed to create an absorption line of a given
    species (f, Gamma, lambda0) as well as max and min values for
    the impact parameter b, redshift z and column density N.

    order_fits: list of str - default: ORDER_FITS
    List of the names of the species in the order that they
    should be fit. Names should correspond to the names of the species
    given in species_dict. (ex: ["lya", "OVI"])

    Return
    ------
    all_species_lines: array
    Array with the parameters of the fitted lines.
    NaNs are placed when encoutering fitting problems

    fit_flux: array
    Fitted flux

    """
    # initialize_output
    dtype = []
    for species in order_fits:
        dtype += [
            (f"{species} N [cm^-2]", float),
            (f"{species} b [km/s]", float),
            (f"{species} zfit", float)
        ]
    all_species_lines = np.array([np.nan], dtype=dtype)

    # load arrays
    if input_extension == ".h5":
        if noise < 0.:
            file = h5py.File(f"{rays_base_name}_spec_nonoise.h5")
        else:
            file = h5py.File(f"{rays_base_name}_spec.h5")
        flux = file['flux'][:]
        wavelength = file["wavelength"][:]
        file.close()
    elif input_extension in [".fits", ".fits.gz"]:
        if noise < 0.:
            hdu = fits.open(f"{rays_base_name}_spec_nonoise{input_extension}")
        else:
            hdu = fits.open(f"{rays_base_name}_spec{input_extension}")
        flux = hdu[1].data["flux"]
        wavelength = hdu[1].data["wavelength"]
        # TODO: read ivar
        hdu.close()
    elif input_extension == ".txt":
        # TODO: read data from txt file
        print(".txt extension not implemented")
        return all_species_lines, np.array([])
    else:
        print("Unknown extension, expected one of .fits, .fits.gz, .txt or .h5")
        return all_species_lines, np.array([])

    # add impact paramenter and redshift guesses
    species_dicts_fit = copy.copy(species_dicts)
    min_z = start_z * Z_MIN_FACTOR
    max_z = start_z * Z_MAX_FACTOR
    min_b = start_b * B_MIN_FACTOR
    max_b = start_b * B_MAX_FACTOR
    for species in species_dicts_fit.values():
        species["max_z"] = max_z
        species["min_z"] = min_z
        species["init_z"] = start_z
        species["max_b"] = max_b
        species["min_b"] = min_b
        species["init_b"] = start_b

    # initialze fit results
    flux_fit = np.ones_like(flux)
    current_chi2 = np.sum((flux_fit - flux)**2)

    # fit all species one at a time in the specified order
    for species in order_fits:

        species_dict = species_dicts_fit.get(species)

        # initialize the fitter class
        leasts_squares = LineProfileLeastSquares(
            wavelength, flux, flux_fit, species_dict)

        # fit line
        minimizer = iminuit.Minuit(
            leasts_squares,
            name=("column_density", "impact_parameter", "redshift"),
            column_density=species_dict["init_N"],
            impact_parameter=species_dict["init_b"],
            redshift=species_dict["init_z"]
        )
        minimizer.limits["column_density"] = (
            species_dict["min_N"], species_dict["max_N"])
        minimizer.limits["impact_parameter"] = (
            species_dict["min_b"], species_dict["max_b"])
        minimizer.limits["redshift"] = (
            species_dict["min_z"], species_dict["max_z"])
        minimizer.errordef = 1.
        minimizer.print_level = 0
        minimizer.migrad()

        # update total fit
        if minimizer.valid and current_chi2 > minimizer.fval:
            minimizer.hesse()

            all_species_lines[f"{species} N [cm^-2]"] = minimizer.values["column_density"]
            all_species_lines[f"{species} b [km/s]"] = minimizer.values["impact_parameter"]
            all_species_lines[f"{species} f"] = minimizer.values["redshift"]

            flux_fit = leasts_squares.compute_line_profile(
                minimizer.values["column_density"],
                minimizer.values["impact_parameter"],
                minimizer.values["redshift"],
            )

            current_chi2 = minimizer.fval

        # fit is now worse, report and skip line
        elif minimizer.valid:
            print(
                f"For spectrum {rays_base_name}, fit on species {species} worsens "
                f"the chi2. Previous chi2 = {current_chi2}."
                f"New chi2 = {minimizer.fval}. Ignoring line")
        # fit did not converge, report and skip line
        else:
            print(
                f"For spectrum {rays_base_name}, fit on species {species} failed. "
                "Ignoring")

    return all_species_lines, flux_fit