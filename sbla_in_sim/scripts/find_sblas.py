"""
This file contains a script to find SBLAs in skewers extracted from simulations.
"""
import argparse
import logging
import multiprocessing
import os
import time

from astropy.table import Table, vstack
import tqdm

from sbla_in_sim.sbla_detection_utils import find_sblas

def process_spectrum(args_tuple):
    """Worker function to process a single spectrum
    
    Arguments
    ---------
    args_tuple: tuple
        (index_spectrum, transmission_file, name, plot)
        
    Returns
    -------
    tuple
        (index_spectrum, sblas_table_all, sblas_table_reduced)
    """
    index_spectrum, transmission_file, name, plot = args_tuple
    
    try:
        sblas_table_all, sblas_table_reduced = find_sblas(transmission_file, name, plot=plot)
        return (index_spectrum, sblas_table_all, sblas_table_reduced)
    except Exception as e:
        # Return the error information so we can handle it in the main process
        return (index_spectrum, None, None, str(e))

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
    
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to use for parallel processing. Default: number of CPU cores."
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
    logger.info(f"Catalogue loaded. Elapsed time: {(t1-t0)} seconds")
    logger.info(f"Number of spectra to analyse: {n_spectra}")

    # Set up multiprocessing
    num_processes = args.num_processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} processes for parallel processing")
    
    # Prepare arguments for parallel processing
    logger.info("Preparing tasks for parallel processing")
    tasks = []
    for index_spectrum in range(n_spectra):
        transmission_file = os.path.join(dir, f"{catalogue['name'][index_spectrum]}_spec_nonoise.fits.gz")
        name = catalogue['name'][index_spectrum]
        tasks.append((index_spectrum, transmission_file, name, args.plot))

    # Process spectra in parallel
    logger.info("Finding SBLAs in spectra using multiprocessing")
    sblas_table_all_list = [None] * n_spectra
    sblas_table_reduced_list = [None] * n_spectra
    errors = []
    
    if num_processes == 1:
        # Use single process with progress bar
        for task in tqdm.tqdm(tasks, desc="Processing spectra"):
            result = process_spectrum(task)
            if len(result) == 4:  # Error case
                index_spectrum, _, _, error = result
                errors.append(f"Spectrum {index_spectrum}: {error}")
            else:
                index_spectrum, sblas_all, sblas_reduced = result
                sblas_table_all_list[index_spectrum] = sblas_all
                sblas_table_reduced_list[index_spectrum] = sblas_reduced
    else:
        # Use multiprocessing
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap for progress tracking
            results = []
            for result in tqdm.tqdm(pool.imap(process_spectrum, tasks), 
                                  total=n_spectra, desc="Processing spectra"):
                results.append(result)
                
            # Process results
            for result in results:
                if len(result) == 4:  # Error case
                    index_spectrum, _, _, error = result
                    errors.append(f"Spectrum {index_spectrum}: {error}")
                else:
                    index_spectrum, sblas_all, sblas_reduced = result
                    sblas_table_all_list[index_spectrum] = sblas_all
                    sblas_table_reduced_list[index_spectrum] = sblas_reduced
    
    # Report any errors
    if len(errors) > 0:
        logger.warning(f"Encountered {len(errors)} errors during processing:")
        for error in errors:
            logger.warning(f"  {error}")
    
        # Filter out None values (from errors)
        sblas_table_all_list = [table for table in sblas_table_all_list if table is not None]
        sblas_table_reduced_list = [table for table in sblas_table_reduced_list if table is not None]
        
    successful_count = len(sblas_table_all_list)
    logger.info(f"Successfully processed {successful_count}/{n_spectra} spectra")

    t2 = time.time()
    logger.info(f"SBLAs found. Elapsed time: {(t2-t1)/60.0} minutes")

    # concatenate all SBLAs
    logger.info("Concatenating SBLA tables")
    sblas_table_all = vstack(sblas_table_all_list)
    sblas_table_reduced = vstack(sblas_table_reduced_list)

    t3 = time.time()
    logger.info(f"SBLA tables concatenated. Elapsed time: {(t3-t2)/60.0} minutes")

    # save tables
    logger.info("Saving SBLA tables")
    sblas_table_all_filename = args.catalogue.replace(".csv","_sblas_all.csv")
    sblas_table_reduced_filename = args.catalogue.replace(".csv","_sblas_reduced.csv")
    if sblas_table_all_filename == args.catalogue:
        raise ValueError("Input catalogue filename must end with .csv to save SBLA tables.")
    if sblas_table_reduced_filename == args.catalogue:
        raise ValueError("Input catalogue filename must end with .csv to save SBLA tables.")    
    sblas_table_all.write(sblas_table_all_filename, overwrite=True)
    sblas_table_reduced.write(sblas_table_reduced_filename, overwrite=True)
    t4 = time.time()
    logger.info(f"SBLA tables saved. Elapsed time: {(t4-t3)/60.0} minutes")

    logger.info(f"Total elapsed time: {(t4-t0)/60.0} minutes")