
from .vis_simulator import simulate_vis
import numpy as np
import healpy as hp

def spectral_index_segments_basic(maps, freqs, boundaries, region_npix=None, 
                                  fill_masked=True, avg_fn=np.mean):
    """
    Split map into regions based on the spectral index calculated between 
    two frequencies. This can result in region with quite disjoint morphologies 
    and complicated boundaries.
    
    Parameters:
        maps (array_like):
            Array of healpix maps, with shape (2, Npix).
        freqs (tuple):
            Pair of frequencies, in Hz, MHz, or GHz.
        boundaries (array_like):
            Boundaries of bins in spectral index.
        region_npix (int):
            If specified, downgrade the resolution of the beta map to 
            simplify the region boundaries, and then upgrade to the 
            original resolution.
        fill_masked (bool):
            If True, fill masked pixels with the mean of the neighbouring 
            pixels of the spectral index map.
        avg_fn (function):
            Which function to use to calculate the representative 
            spectral index of each region. By default, it uses the 
            mean.
    
    Returns:
        beta (array_like):
            Healpix map with the same resolution as the input map, 
            but with integer region IDs as the map values.
    """
    assert maps.shape[0] == 2, "Requires a pair of Healpix maps"
    assert len(freqs) == 2, "Requires a pair of frequencies"
    
    # Make sure boundaries are ordered
    if not np.all(boundaries[:-1] < boundaries[1:]):
        raise ValueError("boundaries array must be sorted in ascending order")
    
    # Calculate per-pixel spectral index map
    beta = np.log(maps[1] / maps[0]) \
         / np.log(freqs[1] / freqs[0])
    
    # Fill masked regions if requested
    if fill_masked:
        idxs = np.where(np.isnan(beta))
        print(idxs)
    
    # Loop over bins
    regions = np.zeros_like(maps[0])
    regions[:] = np.nan
    for i in range(len(boundaries) - 1):
        regions[(beta >= boundaries[i]) & (beta < boundaries[i+1])] = i
    
    # If region_npix is set, downgrade and then upgrade the 
    # resolution to simplify the region shapes
    if region_npix is not None:
        npix_in = hp.npix2nside(maps[0].size)
        r = hp.ud_grade(regions, nside_out=region_npix, pess=False)
        regions = hp.ud_grade(r, nside_out=npix_in, pess=False)
        regions = np.round(regions, decimals=0)
    return regions


def calc_proj_operator_per_region(
    fluxes, region_idxs, ant_pos, antpairs, freqs, times, beams,
    latitude=-0.5361913261514378, multiprocess=True
):
    """
    Calculate a visibility vector for each region of a healpix/healpy map, 
    as a function of frequency, time, and baseline. This is the projection 
    operator from region amplitude to visibilities. Gains are not included.
    
    Parameters:
        fluxes (array_like):
            Flux for each pixel source as a function of frequency. Assumed to 
            be a healpy/healpix array (in Galactic coords) per frequency, with 
            shape `(Npix, Nfreq)`.
        region_idxs (array_like):
            A healpix/healpy map with the integer region ID of each pixel.
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
        multiprocess (bool): Whether to use multiprocessing to speed up the
            calculation
    Returns:
        vis_proj_operator (array_like):
            The projection operator from region amplitudes to visibilities. 
            This is an array of the visibility value contributed by each 
            region if its amplitude were 1.
        unique_regions (array_like):
            An ordered list of integer region IDs.
    """
    unique_regions = np.unique(region_idxs)
    Nregions = unique_regions.size
    Nants = len(ant_pos)
    Nvis = len(antpairs)
    assert fluxes.shape[1] == len(freqs), "`fluxes` must have shape (Npix, Nfreqs)"
    
    # Get pixel Galactic and then equatorial coords of each map pixel (radians)
    nside = hp.npix2nside(re.size)
    theta_gal, phi_gal = hp.pix2ang(nside=nside, ipix=np.arange(m[0].size), lonlat=False)
    theta_eq, phi_eq = hp.Rotator(coord='ge', deg=False)(theta_gal, phi_gal)

    # Empty array of per-point source visibilities
    vis_regions = np.zeros((Nvis, freqs.size, times.size, Nregions), dtype=np.complex128)

    # Get visibility for each region
    for i in range(Nregions):
        
        # Get indices of pixels that are in this region
        idxs = np.where(region_idxs == unique_regions[i])
        
        # Simulate visibility for this region
        # Returns shape (Nfreqs, Ntimes, Nants, Nants)
        vis = simulate_vis(
            ants=ant_pos,
            fluxes=fluxes[idxs,:],
            ra=phi_eq[idxs], # FIXME: Make sure these occupy the correct range!
            dec=theta_eq[idxs],
            freqs=freqs * 1e6,
            lsts=times,
            beams=beams,
            polarized=False,
            precision=2,
            latitude=latitude,
            use_feed="x",
            multiprocess=multiprocess
        )

        # Allocate computed visibilities to only available baselines (saves memory)
        ants = list(ant_pos.keys())
        for j, bl in enumerate(antpairs):
            idx1 = ants.index(bl[0])
            idx2 = ants.index(bl[1])
            vis_regions[j, :, :, i] = vis[:, :, idx1, idx2, i]

    return vis_regions, unique_regions