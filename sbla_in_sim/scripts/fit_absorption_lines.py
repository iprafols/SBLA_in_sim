"""Module to manage fits to the lines"""
import argparse
from itertools import repeat
import logging
import multiprocessing
import os
import time

from astropy.table import Table, hstack
import numpy as np

from sbla_in_sim.config import Config
from sbla_in_sim.line_fit_utils import fit_lines

def main(cmdargs=None):
    """
    Fit an absorption-line spectrum into line profiles.

    Fits the spectrum into absorption complexes and iteratively adds and
    optimizes voigt profiles for each complex.

    Arguments
    ---------
    cmdargs: argparse.Namespace
    Argument parser namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the delta field '
                     'from a list of spectra'))

    parser.add_argument(
        'config_file',
        type=str,
        default=None,
        help=
        ('Configuration file.'))

    args = parser.parse_args(cmdargs)

    t0_0 = time.time()

    # load configuration
    config = Config(args.config_file)
    logger = logging.getLogger("sbla_in_sim")

    #################################
    # continue previous run:        #
    #  - read catalogue             #
    #  - resume skewers computation #
    #################################
    # TODO: figure out how to do this and if it is needed

    ####################################
    # new run:                         #
    #  - read spectra                  #
    #  - fit absorption lines          #
    #  - save results                  #
    ####################################
    logger.info("Starting new run: fitting absorption lines")
    logger.info("Loading catalogue")
    t1_0 = time.time()

    # load catalogue
    catalogue = Table.read(os.path.join(config.output_dir, config.output_catalogue))

    # prepare arguments
    names = [
        f"{config.output_dir}{name}"
        for name in catalogue["name"]
    ]

    context = multiprocessing.get_context('fork')
    with context.Pool(processes=config.num_processors) as pool:
        arguments = zip(
            names,
            repeat(".fits.gz"),
            catalogue["noise"],
            catalogue["z"],
            catalogue["rho"],
        )

        fit_results_list = pool.starmap(fit_lines, arguments)

        # update catalogue
        fit_results = Table(np.concatenate([
            item[0] for item in fit_results_list
        ]))
        catalogue = hstack([catalogue, fit_results])
        
        # save catalogue
        catalogue.write(
            f"{config.output_dir}/{config.catalogue_file}",
            overwrite=True)

    t1_1 = time.time()
    logger.info(f"INFO: Fits done. Elapsed time: {(t1_1-t1_0)/60.0} minutes")
