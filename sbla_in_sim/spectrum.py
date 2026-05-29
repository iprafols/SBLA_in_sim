"""
Spectrum class to generate spectra from rays, with or without noise. 
The class uses trident to generate the spectrum from the ray, and can optionally add noise using a DESI noise model. 
The resulting spectrum can be saved to a fits file for later use.
"""
from astropy.io import fits
import logging
import numpy as np 

import desisim.obs
from desisim.templates import QSO
from desispec.coaddition import coadd_cameras
import trident

from sbla_in_sim.simulate_desi_noise import align, reconstruct_qso_continuum, sim_spectra

logger = logging.getLogger("sbla_in_sim")


LAMBDA_MIN = 3000
LAMBDA_MAX = 9000
LAMBDA_RF_MIN = 1040
LAMBDA_RF_MAX = 1205
DLAMBDA = 0.8
DLAMBDA_HIGHRES = 0.1

QSO_GEN = QSO(minwave=LAMBDA_MIN, maxwave=LAMBDA_MAX, cdelt=DLAMBDA_HIGHRES)

class Spectrum:
    """
    Class to generate spectra from rays, with or without noise.
    
    The class uses trident to generate the spectrum from the ray, and can optionally add noise using a DESI noise model.
    The resulting spectrum can be saved to a fits file for later use.

    Attributes
    ----------
    wave: array of float
    The wavelength array of the spectrum

    flux: array of float
    The flux array of the spectrum

    ivar: array of float
    The inverse variance array of the spectrum

    mean_snr: float
    The mean signal-to-noise ratio of the spectrum in the Lyman-alpha forest region (1040-1205 Angstroms in the rest frame)
    """
    def __init__(self, ray, noise, z_qso=None, mag_qso=None):
        """Generate a spectrum from a ray

        Arguments
        ---------
        ray: trident.Ray
        The ray from which the spectrum will be generated

        noise: bool
        Whether to add noise to the spectrum or not. 
        If True, the spectrum will be generated with noise, otherwise it will be noiseless.

        z_qso: float, optional
        The redshift of the quasar. Required if noise is True.

        mag_qso: float, optional
        The r-band magnitude of the quasar (decam2014-r). Required if noise is True.

        Return
        ------
        spec_gen: trident.SpectrumGenerator
        The generated spectrum generator object with the noiseless spectrum
        """
        self.flux = None
        self.ivar = None
        self.wave = None

        self.mean_snr = None

        if noise:
            self.generate_noisy_spectrum(ray, z_qso, mag_qso)
        else:
            self.generate_noiseless_spectrum(ray)

    def generate_noiseless_spectrum(self, ray):
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
        
        self.wave = spec_gen.lambda_field
        self.flux = spec_gen.flux_field
        self.ivar = np.ones_like(self.flux)
        self.mean_snr = -1.0


    def generate_noisy_spectrum(self, ray, z_qso, mag_qso):
        """Generate a noisy spectrum from a ray

        Arguments
        ---------
        ray: trident.Ray
        The ray from which the spectrum will be generated
        
        z_qso: float
        The redshift of the background quasar
        
        mag_qso: float
        The r-band magnitude of the background quasar (decam2014-r)

        Return
        ------
        spec_gen: trident.SpectrumGenerator
        The generated spectrum generator object with the noisy spectrum
        """
        # generate noiseless quasar spectrum
        # Note: mag_qso should be r-band magnitude (decam2014-r for south=True, BASS-r for south=False)
        try:
            qso_flux, qso_wave, _, obj_meta = QSO_GEN.make_templates(nmodel=1, redshift=[z_qso], mag=[mag_qso])
        except ValueError:
            # Failed to generate template with color cuts - retry without color cuts
            logger.warning(
                f"Failed to generate QSO template for z_qso={z_qso:.5f}, mag_qso={mag_qso:.3f} (r-band) "
                f"with DESI color cuts. Retrying with nocolorcuts=True."
            )
            try:
                qso_flux, qso_wave, _, obj_meta = QSO_GEN.make_templates(
                        nmodel=1, redshift=[z_qso], mag=[mag_qso], nocolorcuts=True
                    )
            except ValueError as e:
                logger.error(
                    f"Failed to generate QSO template for z_qso={z_qso:.5f}, mag_qso={mag_qso:.3f} (r-band) "
                    f"even with nocolorcuts=True."
                )
                raise e

        qso_flux = qso_flux[0] # we only generated one spectrum, so we take the first element of the array
        qso_cont = reconstruct_qso_continuum(qso_flux, qso_wave, z_qso, obj_meta)

        # generate spectrum from the ray
        spec_gen = trident.SpectrumGenerator(
            lambda_min=LAMBDA_MIN,
            lambda_max=LAMBDA_MAX,
            dlambda=DLAMBDA_HIGHRES)
        spec_gen.make_spectrum(
            ray,
            lines='all',
            store_observables=True)
        
        # add the quasar continuum to the spectrum
        final_flux = spec_gen.flux_field * qso_cont

        # simulate DESI noise
        specdata = sim_spectra(qso_wave, final_flux, np.array([z_qso]), exptime=1000, dwave_out=DLAMBDA)
        speccont = sim_spectra(qso_wave, qso_cont, np.array([z_qso]), exptime=1e12, dwave_out=DLAMBDA)

        # align data to the same grid
        common_wave = np.arange(LAMBDA_MIN, LAMBDA_MAX + DLAMBDA, DLAMBDA)
        aligned_specdata = align(specdata, common_wave)
        aligned_speccont = align(speccont, common_wave)

        # coadd cameras and extract the combined spectrum
        coadded_spec = coadd_cameras(aligned_specdata)
        combined_wave = coadded_spec.wave['brz']
        combined_flux = coadded_spec.flux['brz']
        combined_ivar = coadded_spec.ivar['brz']
        coadded_cont = coadd_cameras(aligned_speccont)
        combined_cont = coadded_cont.flux['brz']

        # keep final flux
        self.wave = combined_wave.flatten()
        self.flux = (combined_flux / combined_cont).flatten()
        self.ivar = (combined_ivar * combined_cont**2).flatten()


        rf_wave = combined_wave.flatten() / (1 + z_qso)
        pos = np.where((rf_wave >= LAMBDA_RF_MIN) & (rf_wave <= LAMBDA_RF_MAX))[0]
        self.mean_snr = np.nanmean(combined_flux.flatten()[pos] * np.sqrt(combined_ivar.flatten()[pos]))

    
    def save_spectrum(self, filename):
        """Save the spectrum generated by trident in a fits file

        Arguments
        ---------
        filename: str
        The name of the output file
        """
        header = fits.Header()
        header["WAVE_MIN"] = float(LAMBDA_MIN)
        header["WAVE_MAX"] = float(LAMBDA_MAX)
        header["DWAVE"] = float(DLAMBDA)
        header["MEAN_SNR"] = self.mean_snr
        header["UNITS"] = 'Angs'
        col1 = fits.Column(name='WAVELENGTH', format='E', array=self.wave)
        col2 = fits.Column(name='FLUX', format='E', array=self.flux)
        col3 = fits.Column(name='IVAR', format='E', array=self.ivar)
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols, header=header)

        hdu.writeto(filename, overwrite=True)
        