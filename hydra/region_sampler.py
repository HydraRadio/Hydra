import numpy as np
from .vis_simulator import simulate_vis

import pygdsm
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Galactic, ICRS
import astropy.units as u
import healpy as hp


def get_diffuse_sky_model_pixels(freqs, nside=32, sky_model="gsm2016"):
    """
    Returns arrays of the pixel RA, Dec locations and per-pixel frequency
    spectra from a given sky model. By default this is GSM, as implemented
    in `pygdsm`.

    Parameters:
        freqs (array_like):
            Frequencies, in MHz.

        nside (int):
            Healpix nside to use when constructing the sky model.

        sky_model (str):
            Which sky modle to use, from pyGDSM. One of:
            'gsm2008', 'gsm2016', 'haslam', 'lfss'.

    Returns:
        ra, dec (array_like):
            ICRS RA and Dec locations of each pixel. RA range is (0, 360) deg, 
            Dec range is (-90, +90) deg, but they are returned in radians.

        sky_maps (array_like):
            The frequency spectrum in each pixel. Shape is `(Npix, Nfreq)`.
    """
    # Get expected frequency units
    freqs_MHz = freqs

    # Initialise sky model and extract data cube
    assert sky_model in [
        "gsm2008",
        "gsm2016",
        "haslam",
        "lfsm",
    ], "Available sky models: 'gsm2008', 'gsm2016', 'haslam', 'lfsm'"
    if sky_model == "gsm2008":
        model = pygdsm.GlobalSkyModel(freq_unit="MHz")
    if sky_model == "gsm2016":
        try:
            model = pygdsm.GlobalSkyModel2016(freq_unit="MHz")
        except(AttributeError):
            # Different versions of pygdsm changed the API
            model = pygdsm.GlobalSkyModel16(freq_unit="MHz")
    if sky_model == "haslam":
        model = pygdsm.HaslamSkyModel(freq_unit="MHz")
    if sky_model == "lfsm":
        model = pygdsm.LowFrequencySkyModel(freq_unit="MHz")

    model.generate(freqs_MHz)
    sky_maps = model.generated_map_data  # (Nfreqs, Npix), should be in Kelvin

    # Must change nside to make compute practical
    nside_gsm = hp.npix2nside(sky_maps[0].size)
    sky_maps = hp.ud_grade(sky_maps, nside_out=nside)

    # Convert from Kelvin to Jy/sr
    equiv = u.brightness_temperature(freqs_MHz * u.MHz, beam_area=1 * u.sr)
    Ksr_per_Jy = ((1 * u.Jy).to(u.K, equivalencies=equiv) * u.sr / u.Jy).value
    for i in range(Ksr_per_Jy.size):
        sky_maps[i] /= Ksr_per_Jy[i]  # converts K to Jy/sr

    # Get pixel RA/Dec coords (assumes Galactic coords for sky map data)
    idxs = np.arange(sky_maps[0].size)
    pix_lon, pix_lat = hp.pix2ang(nside, idxs, lonlat=True)
    gal_coords = Galactic(l=pix_lon * u.deg, b=pix_lat * u.deg)

    icrs_frame = ICRS()
    eq_coords = gal_coords.transform_to(icrs_frame)
    ra = eq_coords.ra.rad
    dec = eq_coords.dec.rad

    # Returns list of pixels with spectra as effective point sources
    return ra, dec, sky_maps.T


def segmented_diffuse_sky_model_pixels(
    ra, dec, sky_maps, freqs, nregions, smoothing_fwhm=None
):
    """
    Returns a list of pixel indices for each region in a diffuse sky map.
    This function currently uses a crude spectral index measurement to
    segment the map into regions with roughly equal numbers of pixels.
    Smoothing can be used to reduce sharp edges. The regions can be
    disconnected.

    Parameters:
        ra, dec (array_like):
            ICRS RA and Dec locations of each pixel.

        sky_maps (array_like):
            The frequency spectrum in each pixel.

        freqs (array_like):
            Frequencies, in MHz.

        nregions (int):
            The number of regions of roughly equal numbers of pixels to
            segment the sky map into.

        smoothing_fwhm (float):
            Smoothing FWHM (in degrees) to apply to the segmented map in
            order to reduce sharp edges. The smoothing is applied to a map
            of region indices. It is then re-segmented, which can result
            in a slight reduction in the number of segments due to
            round-off of region indices.

    Returns:
        idxs (list of array_like):
            List of arrays, with each array containing the array indices
            of the pixels that belong to each region.
    """
    # Crude spectral index map
    beta = np.log(sky_maps[:, 0] / sky_maps[:, 1]) / np.log(freqs[0] / freqs[1])

    # Sort spectral index map and break up into ~equal-sized segments
    beta_sorted = np.sort(beta)
    bounds = beta_sorted[:: beta_sorted.size // nregions]

    # Loop over regions and select pixels belonging to each region
    regions = np.zeros(beta.size, dtype=int)
    for i in range(bounds.size - 1):
        # These have >= and <= to ensure that all pixels belong somewhere
        idxs = np.where(np.logical_and(beta >= bounds[i], beta <= bounds[i + 1]))
        regions[idxs] = i

    # Apply smoothing and then re-segment
    if smoothing_fwhm is not None:
        regions_smoothed = hp.smoothing(regions, fwhm=np.deg2rad(smoothing_fwhm))
        regions_final = regions_smoothed.astype(int)
    else:
        regions_final = regions

    # Create list of indices
    unique_idxs = np.sort(np.unique(regions_final))
    idx_list = [np.where(regions_final == i)[0] for i in unique_idxs]
    return idx_list


def calc_proj_operator(
    region_pixel_ra,
    region_pixel_dec,
    region_fluxes,
    region_idxs,
    ant_pos,
    antpairs,
    freqs,
    times,
    beams,
    latitude=-0.5361913261514378,
):
    """
    Calculate a visibility vector for each point source, as a function of
    frequency, time, and baseline. This is the projection operator from point
    source amplitude to visibilities. Gains are not included.

    Parameters:
        region_pixel_ra, region_pixel_dec (list of array_like):
            RA and Dec of each pixel (axcross all regions), in radians.
        region_fluxes (array_like):
            Flux for each pixel (across all regions), as a function of frequency.
        region_idxs (list of array_like):
            List of pixel indices for each region.
        ant_pos (dict):
            Dictionary of antenna positions, [x, y, z], in m. The keys should
            be the numerical antenna IDs.
        antpairs (list of tuple):
            List of tuples containing pairs of antenna IDs, one for each
            baseline.
        freqs (array_like):
            Frequencies, in MHz.
        times (array_like):
            LSTs, in radians.
        beams (list of UVBeam):
            List of UVBeam objects, one for each antenna.
        latitude (float):
            Latitude of the observing site, in radians.

    Returns:
        proj_operator (array_like):
            The projection operator from region amplitudes to visibilities. This
            is an array of the visibility values for each region.
    """
    Nregions = len(region_idxs)
    Nants = len(ant_pos)
    Nvis = len(antpairs)
    ants = list(ant_pos.keys())

    # Empty array of per-point source visibilities
    vis_region = np.zeros((Nvis, freqs.size, times.size, Nregions), dtype=np.complex128)

    # Get visibility for each region
    for j in range(Nregions):
        # Returns shape (NFREQS, NTIMES, NANTS, NANTS)
        vis = simulate_vis(
            ants=ant_pos,
            fluxes=region_fluxes[region_idxs[j], :],
            ra=region_pixel_ra[region_idxs[j]],
            dec=region_pixel_dec[region_idxs[j]],
            freqs=freqs * 1e6,
            lsts=times,
            beams=beams,
            polarized=False,
            precision=2,
            latitude=latitude,
            use_feed="x",
        )

        # Allocate computed visibilities to only available baselines (saves memory)
        for i, bl in enumerate(antpairs):
            idx1 = ants.index(bl[0])
            idx2 = ants.index(bl[1])
            vis_region[i, :, :, j] = vis[:, :, idx1, idx2]

    return vis_region
