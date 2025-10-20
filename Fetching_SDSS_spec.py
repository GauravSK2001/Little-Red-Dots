from pathlib import Path
from astroquery.sdss import SDSS
import astropy.io.fits as fits
from tqdm.notebook import tqdm  # <-- NEW

class SDSSFetcher:
    def __init__(self, data_release=17, outdir="spectra", show_progress=True):
        """
        Initialize the SDSSFetcher class.
        """
        self.data_release = data_release
        self.outdir = Path(outdir)
        self.show_progress = show_progress
        self.outdir.mkdir(exist_ok=True)

    def fetch_spectra(self, id_strings, save=True):
        """
        Download SDSS/BOSS spectra specified by plate-MJD-fiber.
        """
        spectra = []
        # Wrap the iterable in tqdm for progress bar
        for obj in tqdm(id_strings, desc="Fetching SDSS spectra"):
            try:
                plate, mjd, fiber = map(int, obj.split('-'))
            except ValueError:
                print(f"⚠️  Bad format (expected 'plate-mjd-fiber'): {obj}")
                continue

            try:
                spec_list = SDSS.get_spectra(
                    plate=plate,
                    mjd=mjd,
                    fiberID=fiber,
                    data_release=self.data_release,
                    cache=True,
                    show_progress=self.show_progress
                )
            except Exception as exc:
                print(f"⚠️  Network/HTTP error for {obj}: {exc}")
                continue

            if not spec_list:
                print(f"⚠️  No spectrum found for {obj}")
                continue

            spec = spec_list[0]  # keep the HDUList
            spectra.append(spec)

            if save:
                fname = self.outdir / f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"
                spec.writeto(fname, overwrite=True)
                print(f"✅  Saved {fname}")

        return spectra


# Usage example
if __name__ == "__main__":
    file = 'Data/result'
    SDSS_catalogue = fits.open(file)[1].data['sp-id']
    data_release = 12
    outdir = "spectra"
    fetcher = SDSSFetcher(data_release=data_release, outdir=outdir, show_progress=True)
    fetcher.fetch_spectra(SDSS_catalogue[:], save=True)
