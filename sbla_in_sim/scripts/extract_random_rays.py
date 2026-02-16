"""
This file contains a script to run the galaxy simulations using a
uniformly distributed random rays through the 3D volume of the galaxy.
"""
import argparse
from itertools import repeat
import logging
import multiprocessing
import os
import time

import numpy as np
from astropy.table import Table

from sbla_in_sim.config import Config
from sbla_in_sim.random_rays import (
    find_rho_max,
    generate_ray,
    load_snapshot,
    run_simple_ray,
    select_snapshot,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    if config.continue_previous_run: 
        if not os.path.exists(config.output_dir):
            logger.warning(
                "WARNING: continue_previous_run is set to True but "
                "output directory does not exist.")
            logger.warning("Setting continue_previous_run to False")
            config.continue_previous_run = False
        if not os.path.exists(os.path.join(config.output_dir, config.output_catalogue)):
            logger.warning(
                "WARNING: continue_previous_run is set to True but "
                "catalogue file does not exist.")
            logger.warning("Setting continue_previous_run to False")
            config.continue_previous_run = False

    if config.continue_previous_run:

        logger.info("Continuing with existing run")
        logger.info("Loading catalogue")
        t1_0 = time.time()

        # load catalogue
        catalogue = Table.read(os.path.join(config.output_dir, config.output_catalogue))

        # select the entries that were not previously run
        not_run_mask = np.array([
            not (os.path.isfile(os.path.join(config.output_dir, 
                                             entry["name"]+"_spec_nonoise.fits.gz")) 
                 and (entry["noise"] < 0.0 or 
                      os.path.isfile(os.path.join(config.output_dir, 
                                                  entry["name"]+"_spec.fits.gz"))
                      )
                )
            for entry in catalogue
        ])
        not_run_catalogue = catalogue[not_run_mask]
        logger.info(f"{len(not_run_catalogue)} skewer(s) were not previously run")

        # prepare variables to run
        snapshot_names = not_run_catalogue["snapshot_name"]
        redshifts = not_run_catalogue["z"]
        names = not_run_catalogue["name"]
        noise = not_run_catalogue["noise"]
        start_shifts = np.vstack([
            not_run_catalogue["start_shift_x"],
            not_run_catalogue["start_shift_y"],
            not_run_catalogue["start_shift_z"],
        ]).transpose()
        end_shifts = np.vstack([
            not_run_catalogue["end_shift_x"],
            not_run_catalogue["end_shift_y"],
            not_run_catalogue["end_shift_z"],
        ]).transpose()
        galaxy_positions = np.vstack([
            not_run_catalogue["gal_pos_x"],
            not_run_catalogue["gal_pos_y"],
            not_run_catalogue["gal_pos_z"],
        ]).transpose()

        t1_1 = time.time()
        print(f"INFO: Catalogue loaded. Elapsed time: {(t1_1-t1_0)/60.0} minutes")

    ####################################
    # new run:                         #
    #  - compute catalogue and skewers #
    ####################################
    else:
        print("Computing catalogue")
        t1_0 = time.time()
        np.random.seed(config.random_seed)

        # load snapshots info
        snapshots = np.genfromtxt(
            config.snapshots, names=True, dtype=None, encoding="UTF-8")
        snapshots_zmax = np.amax(snapshots["z_max"])

        # generate redshift distributions
        ndz = np.genfromtxt(config.z_dist, names=True, encoding="UTF-8")
        z_from_prob = interp1d(ndz["ndz_pdf"], ndz["z"])
        probs = np.random.uniform(0.0, 1.0, size=config.num_rays)
        redshifts = z_from_prob(probs)
        pos = np.where(redshifts > snapshots_zmax)
        while pos[0].size > 0:
            logger.warning(
                f"{pos[0].size} of the selected redshifts are higher "
                "than the largest snaphot redshift. I will now reassign these "
                "redshifs. This means the redshift distribution will be trimmed. "
                "Consider adding snapshots at larger redshifts or changing the "
                "input redshift distribution")
            probs = np.random.uniform(0.0, 1.0, size=pos[0].size)
            redshifts[pos] = z_from_prob(probs)
            pos = np.where(redshifts > snapshots_zmax)

        # generate random list of starting and ending points for the rays
        snapshots_rho_max = find_rho_max(redshifts, snapshots)
        rho = snapshots_rho_max * np.random.uniform(0, 1, size=config.num_rays)**(1/3)
        theta_e = np.random.uniform(0, 2*np.pi, size=config.num_rays)
        theta_r = np.random.uniform(0, 2*np.pi, size=config.num_rays)
        phi_r = np.random.uniform(-np.pi, np.pi, size=config.num_rays)
        x_start, y_start, z_start, x_end, y_end, z_end = generate_ray(
            rho, theta_e, theta_r, phi_r, 3*snapshots_rho_max)
        start_shifts = np.vstack([x_start, y_start, z_start]).transpose()
        end_shifts = np.vstack([x_end, y_end, z_end]).transpose()

        # generate noise distributions
        if config.noise_dist is not None:
            # draw noises from redshift-dependent distribution
            # Load npz file with histogram2d results (ordering: redshift, mean_snr)
            noise_data = np.load(config.noise_dist)
            H = noise_data['H']  # 2D histogram counts [n_z_bins, n_snr_bins]
            z_edges = noise_data['z_edges']  # redshift bin edges
            snr_edges = noise_data['snr_edges']  # SNR bin edges (noise values)
            
            # Calculate bin centers for SNR
            snr_centers = (snr_edges[:-1] + snr_edges[1:]) / 2
            
            # Sample noise for each ray based on its redshift
            noise = np.zeros_like(redshifts)
            for i, z in enumerate(redshifts):
                # Find which redshift bin this ray falls into
                z_bin = np.digitize(z, z_edges) - 1
                # Clamp to valid range
                z_bin = np.clip(z_bin, 0, H.shape[0] - 1)
                
                # Get the noise distribution for this redshift bin
                snr_distribution = H[z_bin, :]
                
                # Normalize to create probability distribution
                if snr_distribution.sum() > 0:
                    snr_probs = snr_distribution / snr_distribution.sum()
                    # Sample a noise bin according to the distribution
                    noise_bin = np.random.choice(len(snr_centers), p=snr_probs)
                    noise[i] = snr_centers[noise_bin]
                else:
                    # If no data for this redshift bin, use -1.0 (no noise)
                    noise[i] = -1.0
        else:
            noise = np.zeros_like(redshifts) -1.0

        # choose snapshots
        choices = [
            select_snapshot(z_aux, rho_aux, snapshots)
            for z_aux, rho_aux in zip(redshifts, rho)]
        snapshot_names = snapshots["name"][choices]
        galaxy_position_x = snapshots["galaxy_pos_x"][choices]
        galaxy_position_y = snapshots["galaxy_pos_y"][choices]
        galaxy_position_z = snapshots["galaxy_pos_z"][choices]
        galaxy_positions = np.vstack([galaxy_position_x,
                                      galaxy_position_y,
                                      galaxy_position_z]).transpose()

        # get the simulation names
        names = np.array([
            (f"{config.rays_base_name}_{snapshot}_z{z:.4f}_x{xs:.4f}_{xe:.4f}_"
             f"y{ys:.4f}_{ye:.4f}_z{zs:.4f}_{ze:.4f}")
            for snapshot, z, xs, xe, ys, ye, zs, ze in zip(
                snapshot_names,
                redshifts,
                x_start,
                x_end,
                y_start,
                y_end,
                z_start,
                z_end,
            )
        ])

        # save catalogue
        catalogue = Table({
            "name": names,
            "snapshot_name": snapshot_names,
            "z": redshifts,
            "rho": rho,
            "theta_e": theta_e,
            "theta_r": theta_r,
            "start_shift_x": x_start,
            "start_shift_y": y_start,
            "start_shift_z": z_start,
            "end_shift_x": x_end,
            "end_shift_y": y_end,
            "end_shift_z": z_end,
            "gal_pos_x": galaxy_position_x,
            "gal_pos_y": galaxy_position_y,
            "gal_pos_z": galaxy_position_z,
            "noise": noise,
        })
        catalogue.write(os.path.join(config.output_dir, config.output_catalogue))

        t1_1 = time.time()
        logger.info(f"Catalogue created. Elapsed time: {(t1_1-t1_0)/60.0} minutes")

    # run the skewers in parallel
    logger.info("Running skewers using with nproc = %d", config.num_processors)
    t2_0 = time.time()
    for snapshot in np.unique(snapshot_names):
        ds = load_snapshot(snapshot, config.snapshots_dir)
        pos = np.where(snapshot_names == snapshot)
        context = multiprocessing.get_context('fork')
        with context.Pool(processes=config.num_processors) as pool:
            arguments = zip(
                repeat(ds),
                redshifts[pos],
                start_shifts[pos],
                end_shifts[pos],
                galaxy_positions[pos],
                names[pos],
                repeat(config.output_dir),
                noise[pos])

            pool.starmap(run_simple_ray, arguments)


    t2_1 = time.time()
    logger.info(f"Run {len(snapshot_names)} skewers. Elapsed time: {(t2_1-t2_0)/60.0} minutes")

    t0_1 = time.time()
    logger.info(f"Total elapsed time: {(t0_1-t0_0)/60.0} minutes")
