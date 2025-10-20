import numpy as np
import astropy.io.fits as fits
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # progress bars
import msaexp
import matplotlib.pyplot as plt
from numba import jit
# ------------------------------------------------------------------


class SDSS_stacker:
    """
    Stack and re-sample SDSS spectra onto a NIRSpec wavelength grid,
    *including instrumental convolution with uncertainty propagation*.
    """

    # ------------------------------------------------------------------
    # 1. Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        SDSS_catalogue: str,
        SDSS_file_location: str,
        NIRSpec_table_row: dict,
        *,
        max_workers: int | None = None,
        nirspec_R: float | np.ndarray = 2700,
    ) -> None:
        # ---- SDSS master catalogue --------------------------------------
        with fits.open(SDSS_catalogue, memmap=True, lazy_load_hdus=True) as hdul:
            self.SDSS_catalogue = hdul[1].data
        self._z_by_srcid = {str(r["Sp-ID"]): float(r["z"]) for r in self.SDSS_catalogue}

        # ---- SDSS spectra ------------------------------------------------
        self.SDSS_files = glob.glob(f"{SDSS_file_location}/*.fits")

        # ---- NIRSpec reference spectrum ---------------------------------
        url_tpl = "https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}"
        self.NIRSpec = msaexp.spectrum.SpectrumSampler(url_tpl.format(**NIRSpec_table_row)).spec
        self.row = NIRSpec_table_row

        # NIRSpec rest-frame grid (Å) we’ll re-bin everything onto
        self.restframe = self.NIRSpec["wave"] * 1e4 / (1 + self.row["z_best"])

        # NIRSpec instrumental resolution R(λ).  Accept scalar or array.
        if np.ndim(nirspec_R) == 0:
            self.R_nirspec = np.full_like(self.restframe, float(nirspec_R))
        else:
            if len(nirspec_R) != self.restframe.size:
                raise ValueError("nirspec_R array must match NIRSpec wavelength grid length")
            self.R_nirspec = np.asarray(nirspec_R, dtype=float)

        # Thread-pool
        self._max_workers = max_workers

        # Caches
        self.rebinned_spectra: np.ndarray | None = None
        self.rebinned_sigma:   np.ndarray | None = None
        self.stacked:          np.ndarray | None = None

        # Normalised NIRSpec placeholders
        self.NIRSpec_flux_normalized:  np.ndarray | None = None
        self.NIRSpec_sigma_normalized: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 2. Instrument-only convolution & re-sampling
    # ------------------------------------------------------------------
    @jit(nopython=True, cache=True)
    def _resample_template_instr(
        spec_wobs: np.ndarray,
        spec_R_fwhm: np.ndarray | float,
        templ_wobs: np.ndarray,
        templ_flux: np.ndarray,
        templ_unc:  np.ndarray | None = None,
        *,
        nsig: int = 5,
        fill_value: float = np.nan,
    ):
        """
        Degrade *high-resolution* template (`templ_*`) to the instrumental
        resolution described by `spec_R_fwhm` **and** re-sample onto
        `spec_wobs`.  Returns flux + 1-σ uncertainty.
        """
        # Gaussian σ(λ) from instrumental FWHM only
        spec_R_fwhm = np.asarray(spec_R_fwhm, dtype=float)
        dw = spec_wobs / (2.35 * spec_R_fwhm)

        # Template index mapping
        ix  = np.arange(templ_wobs.size)
        ilo = np.interp(spec_wobs - nsig * dw, templ_wobs, ix).astype(int)
        ihi = np.interp(spec_wobs + nsig * dw, templ_wobs, ix).astype(int) + 1

        N    = spec_wobs.size
        fres = np.full(N, fill_value, dtype=float)
        ures = np.full(N, np.nan if templ_unc is None else 0.0)

        for i in range(N):
            sl = slice(ilo[i], ihi[i])
            lam_slice = templ_wobs[sl]

            # Gaussian kernel
            sigma = dw[i]
            g = np.exp(-0.5 * ((lam_slice - spec_wobs[i]) / sigma) ** 2)
            g *= 1.0 / (np.sqrt(2 * np.pi) * sigma)

            # Flux
            fres[i] = np.trapz(templ_flux[sl] * g, lam_slice)

            # Uncertainty
            if templ_unc is not None:
                ures[i] = np.sqrt(np.trapz((templ_unc[sl] ** 2) * (g ** 2), lam_slice))

        return fres, ures

    # Helper to turn SDSS inverse-variance into σ
    @staticmethod
    def _invvar_to_sigma(inv_var: np.ndarray | None):
        if inv_var is None:
            return None
        inv_var = np.asarray(inv_var, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = 1.0 / np.sqrt(inv_var)
        sigma[~np.isfinite(sigma)] = np.nan
        return sigma

    # ------------------------------------------------------------------
    # 3. Re-sample a *single* SDSS spectrum onto the NIRSpec grid
    # ------------------------------------------------------------------
    def _resample_sdss(
        self,
        SDSS_wave: np.ndarray,
        SDSS_flux: np.ndarray,
        z_sdss: float,
        SDSS_uncert: np.ndarray | None = None,
    ):
        """
        Shift an SDSS spectrum into the NIRSpec rest-frame **and**
        convolve it to the NIRSpec instrumental resolution.
        """
        # 3.1  Shift SDSS wavelengths so that λ_rest(SDSS) = λ_rest(NIRSpec)
        #      λ_shifted = λ_SDSS * (1+z_best)/(1+z_sdss)
        templ_wobs_shifted = SDSS_wave * (1 + self.row["z_best"]) / (1 + z_sdss)

        # 3.2  Target grid (observed frame of NIRSpec)
        wave_nirspec_obs = self.restframe * (1 + self.row["z_best"])

        # 3.3  Convolve + re-sample
        if SDSS_uncert is not None:
            SDSS_uncert = self._invvar_to_sigma(SDSS_uncert)

        flux, sig = self._resample_template_instr(
            wave_nirspec_obs,
            self.R_nirspec,
            templ_wobs_shifted,
            SDSS_flux,
            templ_unc=SDSS_uncert,
        )
        return flux, sig, wave_nirspec_obs if SDSS_uncert is not None else flux, wave_nirspec_obs

    # ------------------------------------------------------------------
    # 4. Public method that users previously called "resample"
    # ------------------------------------------------------------------
    def resample(
        self,
        SDSS_wave: np.ndarray,
        SDSS_flux: np.ndarray,
        z_sdss: float,
        SDSS_uncert: np.ndarray | None = None,
    ):
        """
        Back-compatibility wrapper that now calls _resample_sdss and
        keeps the old return signature:
            - with σ : (flux, sigma, wavelength)
            - w/o σ  : (flux, wavelength)
        """
        if SDSS_uncert is not None:
            f, s, w = self._resample_sdss(SDSS_wave, SDSS_flux, z_sdss, SDSS_uncert)
            return f, s, w
        else:
            f, w = self._resample_sdss(SDSS_wave, SDSS_flux, z_sdss)  # type: ignore
            return f, w


    # ------------------------------------------------------------------
    def _normalise_nirspec(self, region: tuple[float, float]):
        """Continuum‑normalise the NIRSpec template (cached)."""
        if self.NIRSpec_flux_normalized is not None:
            return  # already done

        nir_flux  = self.NIRSpec["flux"]  * self.NIRSpec["to_flam"]
        nir_sigma = self.NIRSpec["err"] * self.NIRSpec["to_flam"]

        mask = (self.restframe > region[0]) & (self.restframe < region[1])
        nir_mean = np.nanmean(nir_flux[mask])

        self.NIRSpec_flux_normalized = nir_flux / nir_mean
        self.NIRSpec_flux_normalized[~np.isfinite(self.NIRSpec_flux_normalized)] = np.nan

        sigma_mean = np.sqrt(np.nansum(nir_sigma[mask] ** 2)) / np.sum(mask)
        self.NIRSpec_sigma_normalized = np.sqrt(
            (nir_sigma / nir_mean) ** 2 +
            (self.NIRSpec_flux_normalized * sigma_mean / nir_mean) ** 2
        )
        self.NIRSpec_sigma_normalized[~np.isfinite(self.NIRSpec_sigma_normalized)] = np.nan

    # ------------------------------------------------------------------
    def normalize_flux(
        self,
        SDSS_wave: np.ndarray,
        SDSS_flux: np.ndarray,
        Z_sdss: float,
        SDSS_uncertainty: np.ndarray | None = None,
        *,
        region: tuple[float, float] = (5400, 5600),
    ):
        """Continuum‑normalise and resample an SDSS spectrum."""
        if SDSS_uncertainty is not None:
            res_flux, res_sigma, wave = self.resample(SDSS_wave, SDSS_flux, Z_sdss, SDSS_uncertainty)
        else:
            res_flux, wave = self.resample(SDSS_wave, SDSS_flux, Z_sdss)
            res_sigma = None

        # Continuum level in SDSS rest frame
        rest_wave = wave / (1 + Z_sdss)
        mask = (rest_wave > region[0]) & (rest_wave < region[1])
        if not np.any(mask):
            raise ValueError("Normalisation window contains no pixels!")
        mean_flux = np.nanmean(res_flux[mask])

        norm_flux = res_flux / mean_flux
        norm_flux[~np.isfinite(norm_flux)] = np.nan

        # Ensure NIRSpec is normalised once for this region
        self._normalise_nirspec(region)

        if res_sigma is None:
            return norm_flux

        sigma_mean = np.sqrt(np.nansum(res_sigma[mask] ** 2)) / np.sum(mask)
        norm_sigma = np.sqrt(
            (res_sigma / mean_flux) ** 2 +
            (norm_flux * sigma_mean / mean_flux) ** 2
        )
        norm_sigma[~np.isfinite(norm_sigma)] = np.nan
        return norm_flux, norm_sigma

    # ------------------------------------------------------------------
    def _process_single(self, path: str, *, sig: bool = False):
        """Load one SDSS FITS file and return its normalised, resampled flux."""
        srcid = path.rsplit("/", 1)[-1][5:-5]
        z_sdss = self._z_by_srcid[srcid]
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            data = hdul[1].data
            Lambda = 10.0 ** data["loglam"]
            flux   = data["flux"]
            ivar   = data["ivar"] if sig else None
        if sig:
            return self.normalize_flux(Lambda, flux, z_sdss, ivar)
        else:
            return self.normalize_flux(Lambda, flux, z_sdss)

    # ======================================================================
    # Public API
    # ======================================================================
    def aligning_SDSS_to_NIRSpec(self, *, sig: bool = False, show_progress: bool = True):
        """Re‑bin *all* SDSS spectra onto the NIRSpec grid."""
        if self.rebinned_spectra is not None:
            return self.rebinned_spectra

        worker = lambda p: self._process_single(p, sig=sig)
        desc = "Re‑binning SDSS spectra"
        results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(worker, p) for p in self.SDSS_files]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=desc)
            for fut in iterator:
                results.append(fut.result())

        if sig:
            fluxes, sigmas = zip(*results)
            self.rebinned_spectra = np.asarray(fluxes)
            self.rebinned_sigma   = np.asarray(sigmas)
        else:
            self.rebinned_spectra = np.asarray(results)
        return self.rebinned_spectra

    # ------------------------------------------------------------------
    def Stack(self, *, sig: bool = False):
        """Return the 7‑percentile stack of all rebinned spectra."""
        if self.stacked is not None:
            return self.stacked
        if self.rebinned_spectra is None:
            _ = self.aligning_SDSS_to_NIRSpec(sig=sig)

        percentiles = [0.135, 2.275, 15.865, 50, 84.135, 97.725, 99.865]
        self.stacked = np.nanpercentile(self.rebinned_spectra, percentiles, axis=0)
        return self.stacked

    # ------------------------------------------------------------------
    def save_fits(
        self,
        filename: str,
        *,
        sig: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Write aligned spectra, uncertainties & stack (plus NIRSpec) to FITS."""
        # Ensure everything is computed
        if self.rebinned_spectra is None:
            _ = self.aligning_SDSS_to_NIRSpec(sig=sig)
        if self.stacked is None:
            _ = self.Stack(sig=sig)

        hdr0 = fits.Header()
        hdr0["NSPEC"] = (self.rebinned_spectra.shape[0], "Number of SDSS spectra")
        hdr0["NPIX"]  = (self.restframe.size, "Pixels per spectrum")
        hdr0["ZBEST"] = (self.row["z_best"], "NIRSpec z_best")
        prihdu = fits.PrimaryHDU(header=hdr0)

        wave_hdu   = fits.ImageHDU(self.restframe, name="REST_WAVE")
        align_hdu  = fits.ImageHDU(self.rebinned_spectra, name="ALIGN")
        hdus = [prihdu, wave_hdu, align_hdu]

        if sig and self.rebinned_sigma is not None:
            hdus.append(fits.ImageHDU(self.rebinned_sigma, name="SIGMA"))

        hdus.append(fits.ImageHDU(self.stacked, name="STACK"))

        # Optionally include the normalised NIRSpec template
        if self.NIRSpec_flux_normalized is not None:
            hdus.append(fits.ImageHDU(self.NIRSpec_flux_normalized, name="NIRSPEC_FLUX"))
            hdus.append(fits.ImageHDU(self.NIRSpec_sigma_normalized, name="NIRSPEC_SIGMA"))

        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
    def quick_plot(self,SDSS_='False',Stacked_SDSS='False',LRD_MSAID='False'):
        """Quickly plot the NIRSpec template and SDSS spectra."""
        if self.rebinned_spectra is None:
            _ = self.aligning_SDSS_to_NIRSpec()

        plt.figure(figsize=(12, 6))
        plt.plot(self.restframe, self.NIRSpec_flux_normalized, label="NIRSpec Template", color='black', lw=2)

        if SDSS_:
            for i, spec in enumerate(self.rebinned_spectra):
                plt.step(self.restframe, spec, alpha=0.05,where='mid',color='blue')

        if Stacked_SDSS:
            stack = self.Stack()
            plt.step(self.restframe, stack[3], label="Median Stack", color='black', lw=2)
            plt.fill_between(
                self.restframe,
                stack[1],
                stack[5],
                color='green',
                alpha=0.1,
                label="1 sigma SDSS distribution", step='mid'
            )

        if LRD_MSAID:
            plt.step(self.restframe,self.NIRSpec_flux_normalized, label="LRD MSAID Template", color='red', lw=2,where='mid')
        if ~LRD_MSAID and ~Stacked_SDSS and ~SDSS_:
            return 'No spectra to plot!'
        plt.xlabel("Restframe Wavelength (Å)")
        plt.ylabel("Normalized Flux ($f_{5000 Å}$)")
        plt.legend()
        plt.title("NIRSpec Template and SDSS Spectra")
        plt.grid()
        plt.show()