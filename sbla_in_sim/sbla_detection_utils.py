from astropy.constants import c
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

# list of studied SBLA size (from Muñoz-Santos et al. https://arxiv.org/pdf/2507.08940)
VEL_LIST = sorted( # in km/s
    #[54, 85, 112, 138, 170, 201, 233, 264, 295, 326, 358, 389, 420, 452, 483],
    [85, 112, 138, 170, 201, 233, 264, 295, 326, 358, 389, 420, 452, 483],
    reverse=True)
    
COLOR_LIST = [
    "#E69F00",  # taronja
    "#56B4E9",  # blau cel
    "#009E73",  # verd
    "#F0E442",  # groc
    "#0072B2",  # blau fosc
    "#D55E00",  # vermell terrós
    "#CC79A7",  # magenta
    "#999999",  # gris neutre
    "#117733",  # verd fosc
    "#882255",  # bordeus
    "#44AA99",  # turquesa
    "#DDCC77",  # mostassa
    "#AA4499",  # violeta
    "#88CCEE",  # blau clar
    "#332288",  # blau marí
]
COLOR_DICT = {
    vel: color for vel, color in zip(VEL_LIST, COLOR_LIST)
}

LYA_WL = 1215.67

SBLA_THRESHOLD = 0.25

def find_sblas(transmission_file, name, plot=False):
    """Find SBLAs in a delta_file
    
    Arguments
    ---------
    transmission_file: str
    Path to transmission files in fits format

    name: str
    Name identifier for the spectrum

    plot: boolean - Default: False
    If True, plot the spectra and the found SBLAs.

    Return
    ------
    sblas_table_all: astropy.Table
    Table with all the found SBLAs

    sblas_table_reduced: astropy.Table
    Table with the reduced set of SBLAs
    """
    hdu = fits.open(transmission_file)
    wavelength = hdu[1].data["wavelength"]
    flux = hdu[1].data["flux"]
    weights = 1/(hdu[1].data["flux_error"]**2 + 0.05)
    
    sblas_table_all = Table(
        names=("name", "lambda_abs", "vel_sbla", "lambda_min", "lambda_max", "z"),
        dtype=("U50", ">f4", ">f4", ">f4", ">f4", ">f4"),
    )

    sblas_table_reduced = Table(
        names=("name", "lambda_abs", "vel_sbla", "lambda_min", "lambda_max", "z"),
        dtype=("U50", ">f4", ">f4", ">f4", ">f4", ">f4"),
    )

    # here we will store SBLAs detected for different velocities
    found_sblas_all = {}
    found_sblas_reduced = {}

    # plot the spectrum
    if plot:
        figsize = (10, 5)
        fontsize = 14
        titlefontsize = 8
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
        ax.set_title(
            f"Name: {name}",
            fontsize=titlefontsize
        )
        ax.plot(wavelength, flux)
        xlim = ax.get_xlim()
        ax.hlines(SBLA_THRESHOLD, xlim[0], xlim[1], linestyle="dashed", color="k")
        ax.set_xlim(xlim)
        ax.set_ylabel("Transmitted flux fraction", fontsize=fontsize)
        ax.set_xlabel("Wavelength [Angstrom]", fontsize=fontsize)

    # loop over velocities
    for vel in VEL_LIST:

        # rebin spectrum
        rebin_wave, rebin_delta, rebin_weight, bins_mapping = rebin_spectrum(
            flux, wavelength, weights, vel)

        # find pixels with SBLAs
        pos = np.where(rebin_delta < SBLA_THRESHOLD)
        if pos[0].size < 1:
            continue
    
        # now we iterate over the found pixels to build the SBLA interval
        sblas_list = []
        # first pixel
        selected_bin = pos[0][0] + bins_mapping[0]
        interval = list(wavelength[np.where(bins_mapping == selected_bin)])
        previous_bin = selected_bin
        # subsequent pixels
        for selected_bin in pos[0][1:] + bins_mapping[0]:
            # if pixel is contiguous, then merge intervals
            if selected_bin == previous_bin + 1:
                interval += list(wavelength[np.where(bins_mapping == selected_bin)])
            else:
                # else end current interval
                sblas_list.append(np.array(interval))
                # and start the new one
                interval = list(wavelength[np.where(bins_mapping == selected_bin)])
            previous_bin = selected_bin
        # end the last interval
        sblas_list.append(np.array(interval))
        # keep sblas
        found_sblas_all[vel] = sblas_list
        
        # plot all SBLAs found for this velocity
        if plot:
            item = found_sblas_all[vel][0]
            _, ymax = ax.get_ylim()
            ax.plot(item, [ymax]*item.size, color=COLOR_DICT[vel], label=f"{vel}km/s", linewidth=3)
            for item in found_sblas_all[vel][1:]:
                ax.plot(item, [ymax]*item.size, color=COLOR_DICT[vel], linewidth=3)
    
    # now reduce the number of SBLA so that the larger SBLA eat the smaller ones
    found_sblas_reduced = reduce_intervals(found_sblas_all)
    
    # plot the surviving SBLAs
    if plot:
        for vel, intervals in found_sblas_reduced.items():
            ymin, _ = ax.get_ylim()
            if len(intervals) > 0:
                item = intervals[0]
                ax.plot(item, [ymin]*item.size, color=COLOR_DICT[vel], label=f"surv. {vel}km/s", linewidth=3)
                for item in intervals[1:]:
                    ax.plot(item, [ymin]*item.size, color=COLOR_DICT[vel], linewidth=3)

        fig.savefig(f"{transmission_file.replace('.fits.gz','_sblas.png')}")

    # format full list of SBLAs  into the SBLA table
    for vel, intervals in found_sblas_all.items():
        for interval in intervals:
            lambda_abs = np.mean(interval)
            sblas_table_all.add_row((
                name,
                lambda_abs,
                vel,
                np.min(interval),
                np.max(interval), 
                lambda_abs / LYA_WL - 1,
            ))

    # format reduced list of SBLAs  into the SBLA table
    for vel, intervals in found_sblas_reduced.items():
        for interval in intervals:
            lambda_abs = np.mean(interval)
            sblas_table_reduced.add_row((
                name,
                np.mean(interval),
                vel,
                np.min(interval),
                np.max(interval), 
                lambda_abs / LYA_WL - 1,
            ))

    
    return sblas_table_all, sblas_table_reduced


def rebin_spectrum(delta, wavelength, weight, vel):
    """
    Rebin a spectrum onto a grid of constant velocity spacing.

    Parameters
    ----------
    delta : array_like
        Flux or residuals (same length as wavelength).
    wavelength : array_like
        Wavelength grid of the input spectrum (in same units, e.g. Angstrom).
    weight : array_like
        Weights associated with delta (e.g. inverse variance).
    vel : float
        Desired velocity step in km/s.

    Returns
    -------
    new_wave : ndarray
        Rebinned wavelength grid.
    new_delta : ndarray
        Rebinned delta values (weighted average).
    new_weight : ndarray
        Rebinned weights.
    indexs_mapping : ndarray
        Array of length = len(wavelength), giving for each input pixel
        the index of the new grid bin it was assigned to.
        If outside the rebinned range, value = -1.
    """

    # Speed of light in km/s from Astropy
    c_kms = c.to("km/s").value

    # Convert velocity step to log10 wavelength step
    dloglam = vel / (c_kms * np.log(10))

    # Build new wavelength grid in log space
    loglam = np.log10(wavelength)
    loglam_min, loglam_max = loglam[0], loglam[-1]
    new_loglam = np.arange(loglam_min, loglam_max, dloglam)
    new_wave = 10**new_loglam

    # Bin assignment (0-based)
    inds = np.digitize(loglam, new_loglam) - 1
    inds[(inds < 0) | (inds >= len(new_wave))] = -1
    indexs_mapping = inds.copy()

    # Mask valid points
    valid = inds >= 0
    inds_valid = inds[valid]

    # Weighted sums and total weights per bin (vectorized with bincount)
    sum_w = np.bincount(inds_valid, weights=weight[valid], minlength=len(new_wave))
    sum_dw = np.bincount(inds_valid, weights=delta[valid] * weight[valid], minlength=len(new_wave))

    # Avoid division by zero
    new_weight = sum_w
    new_delta = np.zeros_like(new_wave)
    mask = new_weight > 0
    new_delta[mask] = sum_dw[mask] / new_weight[mask]

    return new_wave, new_delta, new_weight, indexs_mapping

def reduce_intervals(sblas_pos):
    # Sort velocities from largest to smallest
    velocities = sorted(sblas_pos.keys(), reverse=True)
    
    kept = {v: [] for v in velocities}
    larger_intervals = []  # store intervals already accepted from larger velocities
    
    for v in velocities:
        for item in sblas_pos[v]:
            try:
                length_a = item[-1] - item[0]
            except Exception as error:
                print(item)
                print(v)
                print(sblas_pos[v])
                raise error
            eaten = False
            for larger_item in larger_intervals:
                overlap = max(0, min(item[-1], larger_item[-1]) - max(item[0], larger_item[0]))
                if overlap >= 0.5 * length_a:
                    eaten = True
                    break
            if not eaten:
                kept[v].append(item)
                larger_intervals.append(item)
    
    return kept
