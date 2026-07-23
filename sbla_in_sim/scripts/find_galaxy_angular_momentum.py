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
    results: np.ndarray
    A numpy array containing the following:
        - the components of the angular momentum vector (first three elements)
        - the mass of the galaxy (gas)
        - the mass of the galaxy (particles)
    
    units: list
    A list containing the units of the angular momentum vector and mass.
    """
    center = np.array([row['galaxy_pos_x'], row['galaxy_pos_y'], row['galaxy_pos_z']])
    radius = float(row['rho_max']) 
    snapshot_id = int(row['name'])
    ds = yt.load(snapshots_dir + f"/RD{snapshot_id:04d}/RD{snapshot_id:04d}")
    center = ds.arr(center, 'kpc')
    sphere = ds.sphere(center, (radius, 'kpc'))
    bulk_vel = sphere.quantities.bulk_velocity()
    sphere.set_field_parameter("bulk_velocity", bulk_vel)
    angular_momentum = sphere.quantities.angular_momentum_vector()
    mass = sphere.quantities.total_mass()

    results = np.array([
        angular_momentum[0].value, 
        angular_momentum[1].value, 
        angular_momentum[2].value,
        mass[0].value,
        mass[1].value,
    ])
    units = [angular_momentum[0].units, mass[0].units]
    return results, units

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
        default=os.cpu_count()//2 or 1,
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

    if len(results) == 0:
        raise ValueError("Input catalogue is empty; no rows to process.")

    # Each result is (values_array, [angular_momentum_unit, mass_unit]).
    values, units = zip(*results)
    values_array = np.asarray(values)
    angular_momentum_units, mass_units = units[0]

    new_columns = [
        f'angular_momentum_x[{angular_momentum_units}]',
        f'angular_momentum_y[{angular_momentum_units}]',
        f'angular_momentum_z[{angular_momentum_units}]',
        f'mass_gas[{mass_units}]',
        f'mass_particles[{mass_units}]',
    ]

    catalogue[new_columns] = pd.DataFrame(values_array, index=catalogue.index)

    print(f"Saving results to {args.output_file}")
    catalogue.to_csv(args.output_file, sep='\t', index=False)

    t1 = time.time()
    print("Done")
    print(f"Processing time: {t1 - t0:.2f} seconds")