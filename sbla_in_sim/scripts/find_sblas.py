"""
This file contains a script to find SBLAs in skewers extracted from simulations.
"""
import argparse
import logging
import os
import time

from astropy.table import Table
import tqdm

from sbla_in_sim.sbla_detection_utils import find_sblas

def main(cmdargs=None):
    """Run the simulations with rays crossing nearby galaxy centers with a 
    random distribution

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
        'catalogue',
        type=str,
        default=None,
        help=
        ('Catalogue file with the spectra to analyse.'))
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help=
        ('If set, plot the spectra with the found SBLAs.'))
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="File to save the log output."
    )

    args = parser.parse_args(cmdargs)

    #################
    # Logging setup #
    #################
    # Set up logging to always log to terminal, and optionally to file
    logger = logging.getLogger("sbla_in_sim")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    ##################
    # load catalogue #
    ##################
    t0 = time.time()
    logger.info("Loading catalogue")
    dir = os.path.dirname(args.catalogue)
    catalogue = Table.read(args.catalogue)
    n_spectra = len(catalogue)

    t1 = time.time()
    logger.info(f"Catalogue loaded. Elapsed time: {(t1-t0)/60.0} minutes")
    logger.info(f"Number of spectra to analyse: {n_spectra}")

    
    # loop over spectra
    logger.info("Finding SBLAs in spectra")
    sblas_table_all_list = []
    sblas_table_reduced_list = []
    for index_spectrum in tqdm.tqdm(range(n_spectra)):
        transmission_file = os.path.join(dir, f"{catalogue['name'][index_spectrum]}.fits.gz")
        sblas_table_all, sblas_table_reduced = find_sblas(transmission_file, plot=args.plot)

        sblas_table_all_list.append(sblas_table_all)
        sblas_table_reduced_list.append(sblas_table_reduced)

    t2 = time.time()
    logger.info(f"SBLAs found. Elapsed time: {(t2-t1)/60.0} minutes")

    # concatenate all SBLAs
    logger.info("Concatenating SBLA tables")
    sblas_table_all = Table.concatenate(sblas_table_all_list)
    sblas_table_reduced = Table.concatenate(sblas_table_reduced_list)

    t3 = time.time()
    logger.info(f"SBLA tables concatenated. Elapsed time: {(t3-t2)/60.0} minutes")

    # save tables
    logger.info("Saving SBLA tables")
    sblas_table_all.write(args.catalogue.replace(".fits","_sblas_all.fits"), overwrite=True)
    sblas_table_reduced.write(args.catalogue.replace(".fits","_sblas_reduced.fits"), overwrite=True)
    t4 = time.time()
    logger.info(f"SBLA tables saved. Elapsed time: {(t4-t3)/60.0} minutes")

    logger.info(f"Total elapsed time: {(t4-t0)/60.0} minutes")