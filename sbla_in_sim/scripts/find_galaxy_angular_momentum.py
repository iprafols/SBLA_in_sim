"""
This file contains a script to read the catalogue of galaxy positions and sizes and
compute their angular momentum.
"""

import argparse
import time

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
    angular_momentum: np.ndarray
    The angular momentum vector of the galaxy.

    mass: float
    The total mass of the galaxy.
    """
    center = np.array([row['galaxy_pos_x'], row['galaxy_pos_y'], row['galaxy_pos_z']])
    radius = float(row['rho_max']) 
    ds = yt.load(snapshots_dir + f"/RD{row['name']:04d}/RD0{row['name']:04d}")
    center = ds.arr(center, 'kpc')
    sphere = ds.sphere(center, (radius, 'kpc'))
    bulk_vel = sphere.quantities.bulk_velocity()  #es como restarle el mov glov¡bal
    sphere.set_field_parameter("bulk_velocity", bulk_vel)
    angular_momentum = sphere.quantities.angular_momentum_vector()
    mass = sphere.quantities.total_mass()
    return angular_momentum, mass

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
        'input-file',
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
            'If not provided, then use the input file name with "_angular_momentum" 
            'suffix. Note that this assumes the input file has a ".txt" extension.')
    )

    args = parser.parse_args(cmdargs)

    if args.output_file is None:
        args.output_file = args.input_file.replace('.txt', '_angular_momentum.txt')

    t0_0 = time.time()

    catalogue = pd.read_csv(args.input_file, sep='\s+', index_col="name")

    catalogue[['L_x', 'L_y', 'L_z', 'M']] = catalogue.apply(
        find_angular_momentum_and_mass, 
        axis=1,
        result_type='expand',
        arguments=(args.snapshots_dir,),
    )

    catalogue.to_csv(args.output_file, sep='\t', index=False)