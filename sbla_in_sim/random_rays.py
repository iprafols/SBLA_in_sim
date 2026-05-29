"""This file contains functions to extract random rays from a simulation
snapshot"""
import logging
import traceback

from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
import trident
import yt

from sbla_in_sim.spectrum import Spectrum

# some configuration variables used by run_simple_ray
Z_SUN = 0.02041

logger = logging.getLogger("sbla_in_sim")

def find_rho_max(redshifs, snapshots):
    """Find the maximum rho at a given redshift for the specified snapshots

    Arguments
    ---------
    redshif: array of float
    Chosen redshifts

    snapshots: np.structured_array
    A named array with the snapshots information. Must contain fields "name",
    "rho_max" and "z_max"

    Return
    ------
    rho_max: array of float
    The maximum rho
    """
    rho_max = np.zeros_like(redshifs)
    for index, redshif in enumerate(redshifs):
        pos = np.argwhere((snapshots["z_max"] > redshif))
        rho_max[index] = np.max(snapshots["rho_max"][pos])
    return rho_max

def generate_noiseless_spectrum(ray):
    """Generate a noiseless spectrum from a ray

    Arguments
    ---------
    ray: trident.Ray
    The ray from which the spectrum will be generated

    Return
    ------
    spec_gen: trident.SpectrumGenerator
    The generated spectrum generator object with the noiseless spectrum
    """
    spec_gen = trident.SpectrumGenerator(
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX,
        dlambda=DLAMBDA)
    spec_gen.make_spectrum(
        ray,
        lines='all',
        store_observables=True)
    
    return spec_gen

def generate_ray(rho, theta_e, theta_r, phi_r, radius):
    """Function to compute the starting and ending points of a ray

    Arguments
    ---------
    rho: array of float
    Distance between the ray and the centre of the galaxy

    theta_e: array of float
    Angle within the cone of possible rays

    theta_r: array of float
    Rotatiom angle theta for the starting and ending points

    phi_r: array of float
    Rotation angle phi for the starting and ending points

    radius: array of float
    Radius of the sphere including the starting and ending point

    Return
    ------
    x_start: array of float
    x coordinate of the starting point

    y_start: array of float
    y coordinate of the starting point

    z_start: array of float
    z coordinate of the starting point

    x_end: array of float
    x coordinate of the ending point

    y_end: array of float
    y coordinate of the ending point

    z_end: array of float
    z coordinate of the ending point
    """
    # ray start
    x_start, y_start, z_start = rotate(-radius, 0, 0, theta_r, phi_r)

    # ray end
    circle_radius = np.sqrt(4*rho**2-4*rho**4/radius**2)
    x_end_norot = radius - 2*rho**2/radius
    y_end_norot = circle_radius*np.cos(theta_e)
    z_end_norot = circle_radius*np.sin(theta_e)
    x_end, y_end, z_end = rotate(x_end_norot, y_end_norot, z_end_norot, theta_r, phi_r)

    return x_start, y_start, z_start, x_end, y_end, z_end

def load_snapshot(fn, dir=""):
    """Load the simulation snapshot

    Arguments
    ---------
    fn: str
    Name of the snapshot

    dir: str
    Directory where snapshots are kept

    Return
    ------
    ds: yt.data_objects.static_output.Dataset
    The loaded snapshot
    """
    ds = yt.load(f"{dir}RD0{fn}/RD0{fn}")
    ds.add_field(
        ("gas", "metallicity"),
        function=metallicity_e,
        sampling_type="local",
        force_override=True,
        display_name=r"Z/Z$_{\odot}$",
        take_log=True,
        units="")
    return ds

def metallicity_e(field, data):
    """Compute metallicity
    
    Arguments
    ---------
    field: str
    The field name (not used, but required by yt)

    data: yt.data_objects.static_output.Dataset
    The data object containing the fields
    """
    return data["gas", "metal_density"] / data["gas", "density"] / Z_SUN

def rotate(x, y, z, theta, phi):
    """Rotate coordinates around the centre.
    Two rotations are performed following:
    - A rotation on the z axis with angle theta
    - A rotation on the y axis with angle phi

    Arguments
    ---------
    x : float
    x coordinate of the point to rotate

    y : float
    y coordinate of the point to rotate

    z : float
    z coordinate of the point to rotate

    theta: float
    Rotation angle along z axis

    phi: float
    Rotation angle along y axis

    Return
    ------
    x2: float
    x coordinate of the rotated point

    y2: float
    y coordinate of the rotated point

    z2: float
    z coordinate of the rotated point
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # rotation on z axis
    x1 = cos_theta*x - sin_theta*y
    y1 = sin_theta*x + cos_theta*y
    z1 = z

    # rotation on y axis
    x2 = cos_phi*x1 + sin_phi*z1
    y2 = y1
    z2 = -sin_phi*x1 + cos_phi*z1

    return x2, y2, z2

def run_simple_ray(ds,
                   z,
                   start_shift,
                   end_shift,
                   galaxy_pos,
                   base_name,
                   output_dir,
                   z_qso,
                   mag_qso,
                   noise):
    """Run a simple ray from a specified start and end shifts from the centre
    of a galaxy

    Arguments
    ---------
    ds: yt.data_objects.static_output.Dataset
    The loaded snapshot

    z: float
    The redshift of the ray

    start_shift: float, array
    Shift of the starting point of the ray with respect to the galaxy centre
    If a float, all 3 dimensions are equally shifted.
    If an array, it must have size=3 and each dimension will be shifted independently

    end_shift: float, array
    Shift of the ending point of the ray with respect to the galaxy centre.
    If a float, all 3 dimensions are equally shifted.
    If an array, it must have size=3 and each dimension will be shifted independently.

    galaxy: array
    3D position of the galaxy in the snapshot

    base_name: str
    Base name used to name the outputs

    output_dir: str
    Directory where outputs are saved

    z_qso: float
    Redshift of the background quasar

    mag_qso: float
    The magnitude of the background quasar

    noise: bool
    Whether to add noise to the spectrum
    """
    start = galaxy_pos[:] + start_shift
    end = galaxy_pos[:] + end_shift

    datastart = start*ds.units.kpc
    ray_start=datastart.to('code_length')
    dataend = end*ds.units.kpc
    ray_end=dataend.to('code_length')

    ray = trident.make_simple_ray(
        ds,
        start_position=ray_start,
        end_position=ray_end,
        redshift=z,
        lines='all',
        fields=['density', 'temperature', 'metallicity'],
        data_filename=f"{output_dir}{base_name}_ray.h5"
        )
    
    try:
        spec_gen = Spectrum(ray, noise=noise, z_qso=z_qso, mag_qso=mag_qso)
    except ValueError as e:
        logger.error(f"Error generating spectrum for ray {base_name}")
        logger.error(traceback.format_exc())
        logger.error(str(e))
        logger.error("Skiping ray.")
        return


    if noise:
        spec_gen.save_spectrum(f"{output_dir}{base_name}_spec.fits.gz")
    else:
        spec_gen.save_spectrum(f"{output_dir}{base_name}_spec_nonoise.fits.gz")
        
def select_snapshot(redshif, rho, snapshots):
    """Randomly select a snapshot given a choice of z and distance

    Arguments
    ---------
    redshif: float
    Chosen redshift

    rho: float
    Chosen distance

    snapshots: np.structured_array
    A named array with the snapshots information. Must contain fields "name",
    "rho_max" and "z_max"

    Return
    ------
    choice: str
    The position of the chosen
    """
    pos = np.argwhere((snapshots["rho_max"] > rho) & (snapshots["z_max"] > redshif))
    choice = np.random.choice(pos.reshape(pos.shape[0]))
    return choice
