"""Functions to simulate DESI noise on input spectra. The main function is `sim_spectra`, taken from the desisim 
packages and simplified here, which takes as input a wavelength array, a flux array, and a redshift, and returns 
a Spectra object containing the simulated spectra with DESI noise. The function uses the desisim.simexp.simulate_spectra 
function to generate the simulated spectra, and then adds random noise to the fluxes according to the DESI noise model. 
The function also handles padding the input spectra to cover the full DESI wavelength range if necessary."""

from astropy.io import fits
import astropy.units as u
import numpy as np
import os
from scipy.interpolate import interp1d
import time

import desimodel
import desisim
import desispec
from desispec.interpolation import resample_flux
from desispec.spectra import Spectra
from desispec.resolution import Resolution
import desitarget


# desisim templates configuration
template_dir = os.environ['DESI_BASIS_TEMPLATES']
pca_file = os.path.join(template_dir, 'qso_templates_v2.0.fits')
hdu_pca = fits.open(pca_file)
EIGENVECTORS = hdu_pca['SDSS_EIGEN'].data
EIGEN_WAVE = hdu_pca['SDSS_EIGEN_WAVE'].data
hdu_pca.close()

def align(spectra, common_wave):
    """
    Align the wavelength grids of the input spectra to a common grid. This is necessary because the DESI spectrograph has 
    different wavelength grids for each camera (b, r, z), and we want to have a single common grid for all cameras.
    
    Arguments
    ---------
    spectra: desispec.spectra.Spectra
    A Spectra object containing the input spectra to be aligned. The Spectra object should have the same number of spectra 
    for each camera, and the wavelength grids should be in the same order (i.e. b, r, z).

    common_wave: array of float
    The common wavelength grid to which the input spectra will be aligned.
    
    Return
    ------
    aligned_spectra: desispec.spectra.Spectra
    A new Spectra object with the same number of spectra for each camera, but with a common wavelength grid for all cameras. 
    The flux, ivar, and mask arrays will be resampled to the new common grid using linear interpolation for the flux and ivar, 
    and nearest neighbor interpolation for the mask.
    """
    
    nspec = spectra.num_spectra()
    
    # Dictionaries to hold our new aligned data
    new_flux = {}
    new_ivar = {}
    new_mask = {}
    
    for band in spectra.bands:
        new_flux[band] = np.zeros((nspec, len(common_wave)))
        new_ivar[band] = np.zeros((nspec, len(common_wave)))
        new_mask[band] = np.zeros((nspec, len(common_wave)), dtype=np.uint32)
        
        for i in range(nspec):
            # Resample flux and ivar
            f, iv = resample_flux(common_wave, spectra.wave[band], 
                                  spectra.flux[band][i], 
                                  ivar=spectra.ivar[band][i])
            new_flux[band][i] = f
            new_ivar[band][i] = iv
            
            # Resample the mask (nearest neighbor is usually fine for masks)
            # We use interp1d to move the mask to the new grid
            m_interp = interp1d(spectra.wave[band], spectra.mask[band][i], 
                                kind='nearest', bounds_error=False, fill_value=1)
            new_mask[band][i] = m_interp(common_wave).astype(np.uint32)
    
    # Create a NEW Spectra object with aligned grids
    # We pass the same metadata (fibermap) so the object stays valid
    aligned_spectra = Spectra(bands=spectra.bands, 
                              wave={b: common_wave for b in spectra.bands}, 
                              flux=new_flux, 
                              ivar=new_ivar, 
                              mask=new_mask, 
                              fibermap=spectra.fibermap)

    return aligned_spectra

def reconstruct_qso_continuum(qso_flux, qso_wave, qso_z, obj_meta):
    """Reconstruct the quasar continuum from the output of QSO_GEN.make_templates

    Arguments
    ---------
    qso_flux: array of float
    The flux values of the quasar spectrum

    qso_wave: array of float
    The wavelength values of the quasar spectrum

    qso_z: float
    The redshift of the quasar

    obj_meta: dict
    Metadata about the quasar object

    Return
    ------
    qso_cont: array of float
    The reconstructed quasar continuum
    """
    coeffs = obj_meta['PCA_COEFF'][0]

    rest_flux = np.dot(coeffs, EIGENVECTORS)
    wave_obs = EIGEN_WAVE * (1 + qso_z)
    interp_func = interp1d(wave_obs, rest_flux, bounds_error=False, fill_value=0)

    qso_cont = interp_func(qso_wave)

    # Match Normalization
    # make_templates applies a scaling factor to match the r-band magnitude.
    # We need to scale our 'observed_continuum' to match 'qso_flux' in a 
    # region NOT affected by the forest
    # Pick a 'clean' region (1420-1480A rest-frame)
    mask = (qso_wave > 1420 * (1 + qso_z)) & (qso_wave < 1480 * (1 + qso_z))
    scale = np.median(qso_flux[mask] / qso_cont[mask])
    qso_cont *= scale
    
    return qso_cont

def sim_spectra(wave, flux, redshift, exptime, expid=0, seed=0, skyerr=0.0, meta=None, use_poisson=True, dwave_out=None):
    """
    Simulate spectra from an input set of wavelength and flux and writes a FITS file in the Spectra format that can
    be used as input to the redshift fitter.

    Args:
        wave : 1D np.array of wavelength in Angstrom (in vacuum) in observer frame (i.e. redshifted)
        flux : 1D or 2D np.array. 1D array must have same size as wave, 2D array must have shape[1]=wave.size
               flux has to be in units of 10^-17 ergs/s/cm2/A
        redshift : list/array with each index being the redshifts for that target
        exptime: float with each index being the exposure time



    Optional:
        expid : this expid number will be saved in the Spectra fibermap
        seed : random seed
        skyerr : fractional sky subtraction error
        meta : dictionnary, saved in primary fits header of the spectra file
        use_poisson : if False, do not use numpy.random.poisson to simulate the Poisson noise. This is useful to get reproducible random
        realizations.
    """
    if len(flux.shape)==1 :
        flux=flux.reshape((1,flux.size))
    nspec=flux.shape[0]

    print("Starting simulation of {} spectra".format(nspec))
    sourcetype = np.array(["qso" for i in range(nspec)])

    tileid  = 0
    telera  = 0
    teledec = 0
    dateobs = time.gmtime()
    night   = desisim.obs.get_night(utc=dateobs)

    frame_fibermap = desispec.io.fibermap.empty_fibermap(nspec)
    frame_fibermap.meta["FLAVOR"] = "custom"
    frame_fibermap.meta["NIGHT"] = night
    frame_fibermap.meta["EXPID"] = expid

    # add DESI_TARGET
    tm = desitarget.targetmask.desi_mask
    frame_fibermap['DESI_TARGET'] = tm.QSO

    # add TARGETID
    targetid = np.arange(nspec).astype(int)
    frame_fibermap['TARGETID'] = targetid

    # spectra fibermap has two extra fields : night and expid
    # This would be cleaner if desispec would provide the spectra equivalent
    # of desispec.io.empty_fibermap()
    spectra_fibermap = desispec.io.empty_fibermap(nspec)
    spectra_fibermap = desispec.io.util.add_columns(spectra_fibermap,
                       ['NIGHT', 'EXPID', 'TILEID'],
                       [np.int32(night), np.int32(expid), np.int32(tileid)],
                       )

    for s in range(nspec):
        for tp in frame_fibermap.dtype.fields:
            spectra_fibermap[s][tp] = frame_fibermap[s][tp]
    
    obsconditions = desisim.simexp.reference_conditions['DARK']
    obsconditions['EXPTIME'] = exptime
    
    try:
        params = desimodel.io.load_desiparams()
        wavemin = params['ccd']['b']['wavemin']
        wavemax = params['ccd']['z']['wavemax']
    except KeyError:
        wavemin = desimodel.io.load_throughput('b').wavemin
        wavemax = desimodel.io.load_throughput('z').wavemax

    if wave[0] > wavemin:
        print('Minimum input wavelength {}>{}; padding with zeros'.format(wave[0], wavemin))
        dwave = wave[1] - wave[0]
        npad = int((wave[0] - wavemin)/dwave + 1)
        wavepad = np.arange(npad) * dwave
        wavepad += wave[0] - dwave - wavepad[-1]
        fluxpad = np.zeros((flux.shape[0], len(wavepad)), dtype=flux.dtype)
        wave = np.concatenate([wavepad, wave])
        flux = np.hstack([fluxpad, flux])
        assert flux.shape[1] == len(wave)
        assert np.allclose(dwave, np.diff(wave))
        assert wave[0] <= wavemin

    if wave[-1] < wavemax:
        print('Maximum input wavelength {}<{}; padding with zeros'.format(wave[-1], wavemax))
        dwave = wave[-1] - wave[-2]
        npad = int( (wavemax - wave[-1])/dwave + 1 )
        wavepad = wave[-1] + dwave + np.arange(npad)*dwave
        fluxpad = np.zeros((flux.shape[0], len(wavepad)), dtype=flux.dtype)
        wave = np.concatenate([wave, wavepad])
        flux = np.hstack([flux, fluxpad])
        assert flux.shape[1] == len(wave)
        assert np.allclose(dwave, np.diff(wave))
        assert wavemax <= wave[-1]

    ii = (wavemin <= wave) & (wave <= wavemax)

    flux_unit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 )

    wave = wave[ii]*u.Angstrom
    flux = flux[:,ii]*flux_unit

    sim = desisim.simexp.simulate_spectra(wave, flux, fibermap=frame_fibermap,
        obsconditions=obsconditions, redshift=redshift, seed=seed,
        psfconvolve=True, specsim_config_file="desi", dwave_out=dwave_out)

    random_state = np.random.RandomState(seed)
    sim.generate_random_noise(random_state,use_poisson=use_poisson)

    scale=1e17
    specdata = None

    resolution={}
    for camera in sim.instrument.cameras:
        R = Resolution(camera.get_output_resolution_matrix())
        resolution[camera.name] = np.tile(R.to_fits_array(), [nspec, 1, 1])
        resolution[camera.name] = R.to_fits_array()

    skyscale = skyerr * random_state.normal(size=sim.num_fibers)

    for table in sim.camera_output :
        wave = table['wavelength'].astype(float)
        flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
        if np.any(skyscale):
            flux += ((table['num_sky_electrons']*skyscale)*table['flux_calibration']).T.astype(float)

        ivar = table['flux_inverse_variance'].T.astype(float)

        band  = table.meta['name'].strip()[0]

        flux = flux * scale
        ivar = ivar / scale**2
        mask  = np.zeros(flux.shape).astype(int)

        spec = Spectra([band], {band : wave}, {band : flux}, {band : ivar},
                resolution_data=None,
                mask={band : mask},
                fibermap=spectra_fibermap,
                meta=meta,
                single=True)

        if specdata is None :
            specdata = spec
        else :
            specdata.update(spec)

    # need to clear the simulation buffers that keeps growing otherwise
    # because of a different number of fibers each time ...
    desisim.specsim._simulators.clear()
    desisim.specsim._simdefaults.clear()

    return specdata
