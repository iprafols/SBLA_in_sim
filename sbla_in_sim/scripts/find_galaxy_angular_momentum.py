"""
This file contains a script to read the catalogue of galaxy positions and sizes and
compute their angular momentum.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yt


def find_angular_momentum_and_mass(row, snapshots_dir):
    """Find the angular momentum and mass of a galaxy given its position and size.

    Arguments
    ---------
    row: pandas.Series
    A row from the catalogue containing the galaxy's position and size.

    Returns
    -------
    angular_momentum_x: float
    The x-component of the angular momentum vector.
    
    angular_momentum_y: float
    The y-component of the angular momentum vector.

    angular_momentum_z: float
    The z-component of the angular momentum vector.

    mass: float
    The total mass of the galaxy.
    """
    center = np.array([row['galaxy_pos_x'], row['galaxy_pos_y'], row['galaxy_pos_z']])
    radius = float(row['rho_max']) 
    snapshot_id = int(row['name'])
    ds = yt.load(snapshots_dir + f"/RD{snapshot_id:04d}/RD{snapshot_id:04d}")
    center = ds.arr(center, 'kpc')
    sphere = ds.sphere(center, (radius, 'kpc'))
    bulk_vel = sphere.quantities.bulk_velocity()  #es como restarle el mov glov¡bal
    sphere.set_field_parameter("bulk_velocity", bulk_vel)
    angular_momentum = sphere.quantities.angular_momentum_vector()
    mass = sphere.quantities.total_mass()
    return angular_momentum[0], angular_momentum[1], angular_momentum[2], mass

def main(cmdargs=None):
    """Read the catalogue of galaxy positions and sizes and compute their angular momentum.

    Arguments
    ---------
    cmdargs: argparse.Namespace
    Argument parser namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the angular momentum '
                     'from a list of galaxy positions and sizes'))

    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Input file containing galaxy positions and sizes.'
    )

    parser.add_argument(
        '--snapshots-dir',
        type=str,
        default=None,
        help='Directory containing the simulation snapshots.'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help=(
            'Output file to save the computed angular momentum vectors. '
            'If not provided, then use the input file name with "_angular_momentum" '
            'suffix. Note that this assumes the input file has a ".txt" extension.')
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=os.cpu_count() or 1,
        help=(
            'Number of worker processes used to compute angular momentum. '
            'Use 1 to run serially.')
    )

    args = parser.parse_args(cmdargs)

    if args.input_file is None:
        raise ValueError("Input file must be provided using --input-file.")
    if args.snapshots_dir is None:
        raise ValueError("Snapshots directory must be provided using --snapshots-dir.")

    if args.output_file is None:
        args.output_file = args.input_file.replace('.txt', '_angular_momentum.txt')

    t0 = time.time()

    print(f"Reading catalogue from {args.input_file}")
    catalogue = pd.read_csv(args.input_file, sep=r'\s+')

    print("Computing angular momentum vectors and masses")
    records = catalogue.to_dict(orient='records')
    num_workers = max(1, args.num_workers)

    if num_workers == 1:
        results = [find_angular_momentum_and_mass(record, args.snapshots_dir) for record in records]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                executor.map(
                    find_angular_momentum_and_mass,
                    records,
                    [args.snapshots_dir] * len(records),
                    chunksize=1,
                )
            )

    catalogue[['L_x', 'L_y', 'L_z', 'M']] = pd.DataFrame(
        results,
        index=catalogue.index,
    )

    print(f"Saving results to {args.output_file}")
    catalogue.to_csv(args.output_file, sep='\t', index=False)

    t1 = time.time()
    print("Done")
    print(f"Processing time: {t1 - t0:.2f} seconds")