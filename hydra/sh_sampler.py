
try:
    from mpi4py.MPI import SUM as MPI_SUM
except:
    pass

import numpy as np
import scipy as sp
import healpy as hp

from .vis_simulator import simulate_vis_per_alm

# Wigner D matrices
import spherical, quaternionic
import pyuvsim
from scipy.sparse.linalg import cg, LinearOperator

from scipy.stats import invgamma

from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz, ICRS
from astropy.time import Time
import time
import pygdsm


def get_em_ell_idx(lmax):
    """
    With (m,l) ordering!
    """
    ells_list = np.arange(0, lmax + 1)
    em_real = np.arange(0, lmax + 1)
    em_imag = np.arange(1, lmax + 1)
    # ylabel = []

    # First append all real (l,m) values
    Nreal = 0
    i = 0
    idx = []
    ems = []
    ells = []
    for em in em_real:
        for ell in ells_list:
            if ell >= em:
                idx.append(i)
                ems.append(em)
                ells.append(ell)
                Nreal += 1
                i += 1

    # Then all imaginary -- note: no m=0 modes!
    Nimag = 0
    for em in em_imag:
        for ell in ells_list:
            if ell >= em:
                idx.append(i)
                ems.append(em)
                ells.append(ell)
                Nimag += 1
                i += 1
    return ems, ells, idx


def vis_proj_operator_no_rot(
    freqs,
    lsts,
    beams,
    ant_pos,
    lmax,
    nside,
    latitude=-0.5361913261514378,
    include_autos=False,
    autos_only=False,
    ref_freq=100.0,
    spectral_idx=0.0,
):
    """
    Precompute the real and imaginary blocks of the visibility response
    operator. This should only be done once and then "apply_vis_response()"
    is used to get the actual visibilities.

    Parameters:
        freqs (array_like):
            Frequencies, in MHz.
        lsts (array_like):
            LSTs (times) for the simulation. In radians.
        beams (list of pyuvbeam):
            List of pyuveam objects, one for each antenna
        ant_pos (dict):
            Dictionary of antenna positions, [x, y, z], in m. The keys should
            be the numerical antenna IDs.
        lmax (int):
            Maximum ell value. Determines the number of modes used.
        nside (int):
            Healpix nside to use for the calculation (longer baselines should
            use higher nside).
        latitude (float):
            Latitude in decimal format of the simulated array/visibilities.
        include_autos (bool):
            If `True`, the auto baselines are included.
        ref_freq (float):
            Reference frequency for the spectral dependence, in MHz.
        spectral_idx (float):
            Spectral index, `beta`, for the spectral dependence,
            `~(freqs / ref_freq)^beta`.

    Returns:
        vis_response_2D (array_like):
            Visibility operator (δV_ij) for each (l,m) mode, frequency,
            baseline and lst. Shape (Nvis, Nalms) where Nvis is Nbl x Ntimes x Nfreqs.
        ell (array of int):
            Array of ell-values for the visiblity simulation
        m  (array of int):
            Array of ell-values for the visiblity simulation
    """
    ell, m, vis_alm = simulate_vis_per_alm(
        lmax=lmax,
        nside=nside,
        ants=ant_pos,
        freqs=freqs * 1e6,  # MHz -> Hz
        lsts=lsts,
        beams=beams,
        latitude=latitude,
    )

    # Removing visibility responses corresponding to the m=0 imaginary parts
    vis_alm = np.concatenate(
        (vis_alm[:, :, :, :, : len(ell)], vis_alm[:, :, :, :, len(ell) + (lmax + 1) :]),
        axis=4,
    )

    ants = list(ant_pos.keys())
    antpairs = []
    if autos_only == False and include_autos == False:
        auto_ants = []
    for i in ants:
        for j in ants:
            # Toggle via keyword argument if you want to keep the auto baselines/only have autos
            if include_autos == True:
                if j >= i:
                    antpairs.append((ants[i], ants[j]))
            elif autos_only == True:
                if j == i:
                    antpairs.append((ants[i], ants[j]))
            else:
                if j == i:
                    auto_ants.append((ants[i], ants[j]))
                if j > i:
                    antpairs.append((ants[i], ants[j]))

    vis_response = np.zeros(
        (len(antpairs), len(freqs), len(lsts), 2 * len(ell) - (lmax + 1)),
        dtype=np.complex128,
    )

    ## Collapse the two antenna dimensions into one baseline dimension
    # Nfreqs, Ntimes, Nant1, Nant2, Nalms --> Nbl, Nfreqs, Ntimes, Nalms
    for i, bl in enumerate(antpairs):
        idx1 = ants.index(bl[0])
        idx2 = ants.index(bl[1])
        vis_response[i, :] = vis_alm[:, :, idx1, idx2, :]

    # Multiply by spectral dependence model (a powerlaw)
    # Shape: Nbl, Nfreqs, Ntimes, Nalms
    vis_response *= ((freqs / ref_freq) ** spectral_idx)[
        np.newaxis, :, np.newaxis, np.newaxis
    ]

    # Reshape to 2D
    # TODO: Make this into a "pack" and "unpack" function
    # Nbl, Nfreqs, Ntimes, Nalms --> Nvis, Nalms
    Nvis = len(antpairs) * len(freqs) * len(lsts)
    vis_response_2D = vis_response.reshape(Nvis, 2 * len(ell) - (lmax + 1))

    if autos_only == False and include_autos == False:
        autos = np.zeros(
            (len(auto_ants), len(freqs), len(lsts), 2 * len(ell) - (lmax + 1)),
            dtype=np.complex128,
        )
        ## Collapse the two antenna dimensions into one baseline dimension
        # Nfreqs, Ntimes, Nant1, Nant2, Nalms --> Nbl, Nfreqs, Ntimes, Nalms
        for i, bl in enumerate(auto_ants):
            idx1 = ants.index(bl[0])
            idx2 = ants.index(bl[1])
            autos[i, :] = vis_alm[:, :, idx1, idx2, :]

        ## Reshape to 2D
        ## TODO: Make this into a "pack" and "unpack" function
        # Nbl, Nfreqs, Ntimes, Nalms --> Nvis, Nalms
        Nautos = len(auto_ants) * len(freqs) * len(lsts)
        autos_2D = autos.reshape(Nautos, 2 * len(ell) - (lmax + 1))

        return vis_response_2D, autos_2D, ell, m
    else:
        return vis_response_2D, ell, m


def alms2healpy(alms, lmax):
    """
    Takes a real array split as [real, imag] (without the m=0 modes
    imag-part) and turns it into a complex array of alms (positive
    modes only) ordered as in HEALpy.

    Parameters:
        alms (array_like):
            Array of zeros except for the specified mode.
            The array represents all positive (+m) modes including zero
            and has double length, as real and imaginary values are split.
            The first half is the real values.

    Returns:
        healpy_modes (array_like):
            Array of zeros except for the specified mode.
            The array represents all positive (+m) modes including zeroth modes.
    """
    if len(alms.shape) == 1:
        # Combine real and imaginary parts into alm format expected by healpy
        real_imag_split_index = int((np.size(alms) + (lmax + 1)) / 2)
        real = alms[:real_imag_split_index]

        add_imag_m0_modes = np.zeros(lmax + 1) # add m=0 imag. modes back in
        imag = np.concatenate((add_imag_m0_modes, alms[real_imag_split_index:]))
        healpy_modes = real + 1.0j * imag
        return healpy_modes

    elif len(alms.shape) == 2:
        # Handle 2D array case (loop over entries) with a recursion
        return np.array([alms2healpy(modes, lmax) for modes in alms])

    else:
        raise ValueError("alms array must either have shape (Nmodes,) or (Nmaps, Nmodes)")


def healpy2alms(healpy_modes):
    """
    Takes a complex array of alms (positive modes only) and turns into
    a real array split as [real, imag] making sure to remove the
    m=0 modes from the imag-part.

    Parameters:
        healpy_modes (array_like, complex):
            Array of zeros except for the specified mode.
            The array represents all positive (+m) modes including zeroth modes.

    Returns:
        alms (array_like):
            Array of zeros except for the specified mode.
            The array represents all positive (+m) modes including zero
            and is split into a real (first) and imag (second) part. The
            Imag part is smaller as the m=0 modes shouldn't contain and
            imaginary part.
    """
    if len(healpy_modes.shape) == 1:
        # Split healpy mode array into read and imaginary parts, with m=0 
        # imaginary modes excluded (since they are always zero for a real field)
        lmax = hp.sphtfunc.Alm.getlmax(healpy_modes.size)  # to remove the m=0 imag modes
        alms = np.concatenate((healpy_modes.real, healpy_modes.imag[(lmax + 1) :]))
        return alms

    elif len(healpy_modes.shape) == 2:
        # Loop through elements of the 2D input array (recursive) 
        return np.array([healpy2alms(_map) for _map in healpy_modes])

    else:
        raise ValueError("Input array must have shape (Nmodes,) or (Nmaps, Nmodes).")


def get_healpy_from_gsm(
    freq, lmax, nside=64, resolution="low", output_model=False, output_map=False
):
    """
    Generate an array of alms (HEALpy ordered) from gsm 2016
    (https://github.com/telegraphic/pygdsm)

    Parameters:
        freqs (array_like):
            Frequency (in MHz) for which to return GSM model.
        lmax (int):
            Maximum ell value for alms
        nside (int):
            The nside to upgrade/downgrade the map to. Default is nside=64.
        resolution (str):
            if "low/lo/l":  The GSM nside = 64  (default)
            if "hi/high/h": The GSM nside = 1024
        output_model (bool):
            If output_model=True: Outputs model generated from the GSM data.
            If output_model=False (default): no model output.
        output_map (bool):
            If output_map=True: Outputs map generated from the GSM data.
            If output_map=False (default): no map output.

    Returns:
        healpy_modes (array_like):
            Complex array of alms with same size and ordering as in healpy (m,l)
        gsm_2016 (PyGDSM 2016 model):
            If output_model=True: Outputs model generated from the GSM data.
            If output_model=False (default): no model output.
        gsm_map (healpy map):
            If output_map=True: Outputs map generated from the GSM data.
            If output_map=False (default): no map output.

    """
    # Instantiate GSM model and extract alms
    try:
        gsm_2016 = pygdsm.GlobalSkyModel2016(freq_unit="MHz", resolution=resolution)
    except(AttributeError):
        gsm_2016 = pygdsm.GlobalSkyModel16(freq_unit="MHz", resolution=resolution)
    
    gsm_map = gsm_2016.generate(freqs=freq)
    gsm_upgrade = hp.ud_grade(gsm_map, nside)
    healpy_modes_gal = np.array([hp.map2alm(maps=_map, lmax=lmax) for _map in gsm_upgrade])

    # By default it is in gal-coordinates, convert to equatorial
    rot_gal2eq = hp.Rotator(coord="GC")
    healpy_modes_eq = np.array([rot_gal2eq.rotate_alm(_modes) for _modes in healpy_modes_gal])

    if output_model == False and output_map == False:  # default
        return healpy_modes_eq
    elif output_model == False and output_map == True:
        return healpy_modes_eq, gsm_map
    elif output_model == True and output_map == False:
        return healpy_modes_eq, gsm_2016
    else:
        return healpy_modes_eq, gsm_2016, gsm_map


def get_alms_from_gsm(
    freq, lmax, nside=64, resolution="low", output_model=False, output_map=False
):
    """
    Generate a real array split as [real, imag] (without the m=0 modes
    imag-part) from gsm 2016 (https://github.com/telegraphic/pygdsm)

    Parameters:
    freqs (float or array_like):
        Frequency (in MHz) for which to return GSM model
    lmax (int):
        Maximum ell value for alms
    nside (int):
        The nside to upgrade/downgrade the map to. Default is nside=64.
    resolution (str):
        if "low/lo/l":  nside = 64  (default)
        if "hi/high/h": nside = 1024
    output_model (bool):
        If output_model=True: Outputs model generated from the GSM data.
        If output_model=False (default): no model output.
    output_map (bool):
        If output_map=True: Outputs map generated from the GSM data.
        If output_map=False (default): no map output.

    Returns:
        alms (array_like):
            Array of zeros except for the specified mode.
            The array represents all positive (+m) modes including zero
            and has double length, as real and imaginary values are split.
            The first half is the real values.
        gsm_2016 (PyGDSM 2016 model):
            If output_model=True: Outputs model generated from the GSM data.
            If output_model=False (default): no model output.
        gsm_map (healpy map):
            If output_map=True: Outputs map generated from the GSM data.
            If output_map=False (default): no map output.
    """
    return healpy2alms(
        get_healpy_from_gsm(freq, lmax, nside, resolution, output_model, output_map)
    )


def construct_rhs_no_rot(
    data, inv_noise_var, inv_prior_var, omega_0, omega_1, a_0, vis_response
):
    """
    Construct RHS of linear system.
    """
    real_data_term = vis_response.real.T @ (
        inv_noise_var * data.real + np.sqrt(inv_noise_var) * omega_1.real
    )
    imag_data_term = vis_response.imag.T @ (
        inv_noise_var * data.imag + np.sqrt(inv_noise_var) * omega_1.imag
    )
    prior_term = inv_prior_var * a_0 + np.sqrt(inv_prior_var) * omega_0

    right_hand_side = real_data_term + imag_data_term + prior_term

    return right_hand_side


def apply_lhs_no_rot(a_cr, inv_noise_var, inv_prior_var, vis_response):
    """
    Apply LHS operator of linear system to an input vector.
    """
    real_noise_term = (
        vis_response.real.T @ (inv_noise_var[:, np.newaxis] * vis_response.real) @ a_cr
    )
    imag_noise_term = (
        vis_response.imag.T @ (inv_noise_var[:, np.newaxis] * vis_response.imag) @ a_cr
    )
    signal_term = inv_prior_var * a_cr

    left_hand_side = real_noise_term + imag_noise_term + signal_term
    return left_hand_side


def construct_rhs_no_rot_mpi(
    comm, data, inv_noise_var, inv_prior_var, omega_a, omega_n, a_0, vis_response
):
    """
    Construct RHS of linear system from data split across multiple MPI workers.
    """
    if comm is not None:
        myid = comm.Get_rank()
    else:
        myid = 0

    # Synchronise omega_a across all workers
    if myid != 0:
        omega_a *= 0.
    if comm is not None:
        comm.Bcast(omega_a, root=0)

    # Calculate data terms
    my_data_term = vis_response.real.T @ (
        (inv_noise_var * data.real).flatten()
        + np.sqrt(inv_noise_var).flatten() * omega_n.real.flatten()
    ) + vis_response.imag.T @ (
        (inv_noise_var * data.imag).flatten()
        + np.sqrt(inv_noise_var).flatten() * omega_n.imag.flatten()
    )

    # Do Reduce (sum) operation to get total operator on root node
    data_term = np.zeros(
        (1,), dtype=my_data_term.dtype
    )  # dummy data for non-root workers
    if myid == 0:
        data_term = np.zeros_like(my_data_term)
    
    if comm is not None:
        comm.Reduce(my_data_term, data_term, op=MPI_SUM, root=0)
        comm.barrier()
    else:
        data_term = my_data_term

    # Return result (only root worker has correct result)
    if myid == 0:
        return data_term + inv_prior_var * a_0 + np.sqrt(inv_prior_var) * omega_a
    else:
        return np.zeros_like(a_0)


def apply_lhs_no_rot_mpi(comm, a_cr, inv_noise_var, inv_prior_var, vis_response):
    """
    Apply LHS operator of linear system to an input vector that has been
    split into chunks between MPI workers.
    """
    if comm is not None:
        myid = comm.Get_rank()
    else:
        myid = 0

    # Synchronise a_cr across all workers
    if myid != 0:
        a_cr *= 0.
    if comm is not None:
        comm.Bcast(a_cr, root=0)

    # Calculate noise terms for this rank
    my_tot_noise_term = (
        vis_response.real.T
        @ (inv_noise_var.flatten()[:, np.newaxis] * vis_response.real)
        @ a_cr
        + vis_response.imag.T
        @ (inv_noise_var.flatten()[:, np.newaxis] * vis_response.imag)
        @ a_cr
    )

    # Do Reduce (sum) operation to get total operator on root node
    tot_noise_term = np.zeros(
        (1,), dtype=my_tot_noise_term.dtype
    )  # dummy data for non-root workers
    if myid == 0:
        tot_noise_term = np.zeros_like(my_tot_noise_term)
    
    if comm is not None:
        comm.Reduce(my_tot_noise_term, tot_noise_term, op=MPI_SUM, root=0)
    else:
        tot_noise_term = my_tot_noise_term

    # Return result (only root worker has correct result)
    if myid == 0:
        signal_term = inv_prior_var * a_cr
        return tot_noise_term + signal_term
    else:
        return np.zeros_like(a_cr)


def radiometer_eq(
    auto_visibilities, ants, delta_time, delta_freq, Nnights=1, include_autos=False
):
    nbls = len(ants)
    indx = auto_visibilities.shape[0] // nbls

    sigma_full = np.empty((0))  # , autos.shape[-1]))

    for i in ants:
        vis_ii = auto_visibilities[i * indx : (i + 1) * indx]  # ,:]

        for j in ants:
            if include_autos == True:
                if j >= i:
                    vis_jj = auto_visibilities[j * indx : (j + 1) * indx]  # ,:]
                    sigma_ij = (vis_ii * vis_jj) / (Nnights * delta_time * delta_freq)
                    sigma_full = np.concatenate((sigma_full, sigma_ij))
            else:
                if (
                    j > i
                ):  # only keep this line if you don't want the auto baseline sigmas
                    vis_jj = auto_visibilities[j * indx : (j + 1) * indx]  # ,:]
                    sigma_ij = (vis_ii * vis_jj) / (Nnights * delta_time * delta_freq)
                    sigma_full = np.concatenate((sigma_full, sigma_ij))

    return sigma_full


def sample_cl(alms, ell, m):
    """
    Sample C_ell from an inverse gamma distribution, given a set of
    SH coefficients. See Eq. 7 of Eriksen et al. (arXiv:0709.1058).
    """
    # Get m, ell ordering
    m_vals, ell_vals, lm_idxs = get_em_ell_idx(lmax)

    # Calculate sigma_ell = 1/(2 l + 1) sum_m |a_lm|^2
    for ell in np.unique(ell_vals):
        idxs = np.where(ell_vals == ell)

    #sigma_ell = 
    #x = invgamma.rvs(loc=1, scale=1)
    #C_l = x ((2l+1)/2) sigma_l
    #a = (2l-1)/2
