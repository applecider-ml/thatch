# Spectroscopy

THATCH extracts spectra from two HST spectroscopic modes:

## STIS Slit Spectroscopy

For STIS observations, THATCH uses the pipeline-extracted 1D spectra (x1d/sx1 FITS files) directly from MAST. These are flux-calibrated by the `calstis` pipeline.

**Supported gratings**: G140L, G230L, G230LB, G430L, G750L (R ~ 500–10,000)

### SN 2011fe: 71 STIS Spectra

THATCH extracted 71 STIS spectra spanning days -14 to +3996 relative to B-maximum, covering UV through optical wavelengths:

![SN 2011fe spectra](../images/SN2011fe_spectral_sequence.png)

*HST/STIS spectral sequence of SN 2011fe showing evolution from pre-maximum through the nebular phase. UV (G230LB), optical (G430L), and red (G750L) gratings are combined at each epoch.*

## WFC3/IR Grism Spectroscopy

For slitless grism data (G102, G141), THATCH uses [grizli](https://github.com/gbrammer/grizli) for forward-model extraction with contamination subtraction. The workflow:

1. Pair each grism FLT with its contemporaneous direct image FLT
2. Build a segmentation map at the transient position
3. Initialize grizli's `GrismFLT` model with grism trace calibrations
4. Compute the contamination model and perform optimal 1D extraction
5. Flux-calibrate using the grism sensitivity curve

**Supported grisms**: G102 (0.8–1.15 μm, R~210), G141 (1.1–1.7 μm, R~130)

### AT2017gfo: Grism Extraction

THATCH extracted flux-calibrated G102+G141 spectra at 4 epochs (+5, +7, +9, +10 days post-merger):

![AT2017gfo grism](../images/AT2017gfo_grism_grizli.png)

*WFC3/IR grism spectra of AT2017gfo extracted with grizli. Each epoch combines 4 dithered exposures.*

## Combined UV-to-NIR Spectra

By combining STIS UV spectroscopy with WFC3/IR grism data, THATCH produces continuous 0.15–1.7 μm spectral coverage from HST alone:

![AT2017gfo full spectrum](../images/AT2017gfo_full_spectrum_grizli.png)

*AT2017gfo at ~5 days post-merger: STIS G230L (UV) + WFC3 G102 + G141 (NIR), with X-shooter ground-based spectrum for comparison.*

## Output Format

Spectra are stored in HDF5, with each spectrum as a group containing:

- `wavelength` — array in Angstroms
- `flux` — array in erg/s/cm²/Å
- `flux_err` — array in erg/s/cm²/Å
- Metadata attributes: `mjd`, `instrument`, `grating`, `exptime_s`, `proposal_id`
