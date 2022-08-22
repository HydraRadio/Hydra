import numpy as np
import numpy.fft as fft

import vis_cpu
import pyuvsim
import time

from .vis_simulator import simulate_vis_per_source
from scipy.sparse.linalg import cg, LinearOperator


def precompute_op(vis_proj_operator, inv_noise_var):
    """
    Precompute the real and imaginary blocks of the matrix operator
    (v^T N^-1 v), which is the most expensive component of the LHS operator of
    the linear system.

    Parameters:
        vis_proj_operator (array_like):
            The projection operator from source amplitudes to visibilities,
            from `calc_proj_operator`.
        inv_noise_var (array_like):
            Inverse noise variance array, with the same shape as the visibility
            data.

    Returns:
        vsquared (array_like):
            The sum of the real and imaginary blocks of the matrix operator
            (v^T_re N^-1 v_re + v^T_im N^-1 v_im), which has shape
            (Nsrcs, Nsrcs).
    """
    nsrcs = vis_proj_operator.shape[-1]
    v = vis_proj_operator * np.sqrt(inv_noise_var)[:, :, :, np.newaxis]
    v = v.reshape((-1, nsrcs))
    return v.T.real @ v.real + v.T.imag @ v.imag


def apply_operator(x, amp_prior_std, vsquared):
    """
    Apply LHS operator to a vector of source amplitudes.

    Parameters:
        x (array_like):
            Vector of source amplitudes to apply the operator to.
        amp_prior_std (array_like):
            Vector of standard deviation values to use for independent Gaussian
            priors on the source amplitudes.
        vsquared (array_like):
            Sum of the real and imaginary blocks of the precomputed matrix
            (v^T N^-1 v), which is the most expensive part of the LHS operator.
            Precomputing this typically leads to a large speed-up. The
            `precompute_op` function generates the necessary operator.

    Returns:
        lhs (array_like):
            Result of applying the LHS operator to the input vector, x.
    """
    return x + amp_prior_std * (vsquared @ (x * amp_prior_std))


def construct_rhs(
    resid, inv_noise_var, amp_prior_std, vis_proj_operator, realisation=False
):
    """
    Construct the RHS vector of the linear system. This will have shape
    (2*Nsrcs), as the real and imaginary parts are separated.

    Parameters:
        resid (array_like):
            Residual of data minus reference model, with the same shape as the
            visibility data.
        inv_noise_var (array_like):
            Inverse noise variance array, with the same shape as the visibility
            data.
        amp_prior_std (array_like):
            Vector of standard deviation values to use for independent Gaussian
            priors on the source amplitudes.
        vis_proj_operator (array_like):
            The projection operator from source amplitudes to visibilities,
            from `calc_proj_operator`.
        realisation (bool):
            Whether to include the random realisation terms in the RHS
            (constrained realisation), or just the deterministic terms (Wiener
            filter).

    Returns:
        rhs (array_like):
            The RHS of the linear system.
    """
    Nptsrc = amp_prior_std.size
    proj = vis_proj_operator.reshape((-1, Nptsrc))

    # Switch to turn random realisations on or off
    realisation_switch = 1.0 if realisation else 0.0

    # (Term 2): \omega_a
    b = realisation_switch * np.random.randn(Nptsrc) # real vector

    # (Terms 1+3): S^1/2 A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    omega_n = (
        realisation_switch
        * (1.0 * np.random.randn(*resid.shape) + 1.0j * np.random.randn(*resid.shape))
        / np.sqrt(2.0)
    )

    # Separate complex part of RHS into real and imaginary parts, and apply
    # the real and imaginary parts of the projection operator separately.
    # This is necessary to get a real RHS vector
    y = ((resid * inv_noise_var) + (omega_n * np.sqrt(inv_noise_var))).flatten()
    b += amp_prior_std * (proj.T.real @ y.real + proj.T.imag @ y.imag)
    return b


def calc_proj_operator(
    ra, dec, fluxes, ant_pos, antpairs, freqs, times, beams,
    latitude=-0.5361913261514378
):
    """
    Calculate a visibility vector for each point source, as a function of
    frequency, time, and baseline. This is the projection operator from point
    source amplitude to visibilities. Gains are not included.

    Parameters:
        ra, dec (array_like):
            RA and Dec of each source, in radians.
        fluxes (array_like):
            Flux for each point source as a function of frequency.
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
        vis_proj_operator (array_like):
            The projection operator from source amplitudes to visibilities,
            from `calc_proj_operator`. This is an array of the visibility
            values for each source.
    """
    Nptsrc = ra.size
    Nants = len(ant_pos)
    Nvis = len(antpairs)

    # Empty array of per-point source visibilities
    vis_ptsrc = np.zeros((Nvis, freqs.size, times.size, Nptsrc), dtype=np.complex128)

    # Get visibility for each point source
    # Returns shape (NFREQS, NTIMES, NANTS, NANTS, NSRCS)
    vis = simulate_vis_per_source(
        ants=ant_pos,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs * 1e6,
        lsts=times,
        beams=beams,
        polarized=False,
        precision=2,
        latitude=latitude,
        use_feed="x",
    )

    # Allocate computed visibilities to only available baselines (saves memory)
    ants = list(ant_pos.keys())
    for i, bl in enumerate(antpairs):
        idx1 = ants.index(bl[0])
        idx2 = ants.index(bl[1])
        vis_ptsrc[i, :, :, :] = vis[:, :, idx1, idx2, :]

    return vis_ptsrc
