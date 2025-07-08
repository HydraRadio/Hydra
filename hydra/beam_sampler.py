import numpy as np
from scipy.linalg import toeplitz, cholesky, LinAlgError, solve
from scipy.special import comb, hyp2f1, jn_zeros, jn

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from pyuvdata.analytic_beam import UniformBeam

from .sparse_beam import sparse_beam
from .vis_simulator import simulate_vis_per_source
from . import utils


def split_real_imag(
        arr, 
        kind
    ): # TODO: Should this be a utility function instead of hiding in beam_sampler?
    """
    Split an array into real and imaginary components in different ways
    depending on whether the array is an operator or a vector.

    Parameters:
        arr (array_like):
            The array for which the components are to be split.
        kind (str):
            The role that the array plays in the linear algebra. Either 'op'
            (for operator) or 'vec' (for vector).

    Returns:
        new_arr (array_like):
            Array that has been split into real and imaginary parts.
    """
    valid_kinds = ["op", "vec"]
    nax = len(arr.shape)
    if kind not in valid_kinds:
        raise ValueError(
            "kind must be 'op' or 'vec' when splitting complex "
            "arrays into real and imaginary components"
        )
    if kind == "op":
        new_arr = np.zeros((2, 2) + arr.shape, dtype=float)
        new_arr[0, 0] = arr.real
        new_arr[0, 1] = -arr.imag
        new_arr[1, 0] = arr.imag
        new_arr[1, 1] = arr.real

        # Prepare to put these axes at the end
        axes = list(range(2, nax + 2)) + [0, 1]

    elif kind == "vec":
        new_arr = np.zeros((2,) + arr.shape, dtype=float)
        new_arr[0], new_arr[1] = (arr.real, arr.imag)

        # Prepare to put this axis at the end
        axes = list(range(1, nax + 1)) + [
            0,
        ]

    # Rearrange axes (they are always expected at the end)
    new_arr = np.transpose(new_arr, axes=axes)
    return new_arr


def reshape_data_arr(
        arr, 
        Nfreqs, 
        Ntimes, 
        Nants, 
        Npol
    ):
    """
    Reshape a data-shaped array into a Hermitian matrix representation that is
    more convenient for beam sampling. Makes a copy of the data twice as large
    as the data.

    Parameters:
        arr (array_like, complex):
            Array to be reshaped.
        Nfreqs (int):
            Number of frequencies in the data.
        Ntimes (int):
            Number of times in the data.
        Nants (int):
            Number of antennas in the data.
        Npol (int):
            Number of polarizations per antenna (2 if polarized, 1 if not)

    Returns:
        arr_beam (array_like, complex):
            The reshaped array
    """

    arr_trans = np.transpose(arr, (0, 1, 3, 4, 2))
    arr_beam = np.zeros(
        [Npol, Npol, Nfreqs, Ntimes, Nants, Nants], dtype=arr_trans.dtype
    )
    for pol_ind1 in range(Npol):
        for pol_ind2 in range(Npol):
            for freq_ind in range(Nfreqs):
                for time_ind in range(Ntimes):
                    triu_inds = np.triu_indices(Nants, k=1)
                    arr_beam[
                        pol_ind1,
                        pol_ind2,
                        freq_ind,
                        time_ind,
                        triu_inds[0],
                        triu_inds[1],
                    ] = arr_trans[pol_ind1, pol_ind2, freq_ind, time_ind]

    return arr_beam


def get_bess_matr(
        nmodes, 
        mmodes, 
        rho, 
        phi
    ):
    """
    Make the matrix that evaluates the sparse Fourier-Bessel basis at
    source positions in a polar projection defined by rho and phi.

    Parameters:
        nmodes (array_like):
            Radial modes to use for the Bessel basis. Should correspond to mmodes
            argument.
        mmodes (array_like):
            Which azimuthal modes to use for the Fourier-Bessel basis
        rho (array_like):
            Radial coordinate on disc (usually a monotonic function of radio astronomer's zenith angle).
        phi (array_like):
            Azimuthal coordinate on disc (usually radio astronomer's azimuth angle)

    Returns:
        bess_matr (array_like):
            Fourier-Bessel transformation matrix.
    """

    unique_n, ninv = np.unique(nmodes, return_inverse=True)
    nmax = np.amax(unique_n)
    bzeros = jn_zeros(0, nmax)
    bess_norm = jn(1, bzeros)
    # Shape Ntimes, Nptsrc, nmax
    bess_modes = jn(0, bzeros[np.newaxis, np.newaxis, :] * rho[:, :, np.newaxis])
    # Shape Ntimes, Nptsrc, len(nmodes)
    bess_vals = bess_modes[:, :, ninv] / bess_norm[ninv]

    unique_m, minv = np.unique(mmodes, return_inverse=True)
    # Shape Ntimes, Nptsrc, len(unique_m)
    az_modes = np.exp(
        1.0j * unique_m[np.newaxis, np.newaxis, :] * phi[:, :, np.newaxis]
    )
    # Shape Ntimes, Nptsrc, len(mmodes)
    az_vals = az_modes[:, :, minv] / np.sqrt(np.pi)  # making orthonormal

    bess_matr = bess_vals * az_vals

    return bess_matr


def fit_bess_to_beam(
    beam, 
    freqs, 
    nmodes, 
    mmodes, 
    rho, 
    phi, 
    polarized=False, 
    force_spw_index=False
):
    """
    Get the least-squares fit Fourier-Bessel coefficients for a beam based on 
    its value at a set of points in a polar projection defined by rho and phi.

    Parameters:
        beam (pyuvsim.Beam):
            The beam being fitted.
        freqs (array_like):
            Frequencies for the beam, in Hz.
        nmodes (array_like):
            Radial modes to use for the Bessel basis. Should correspond to mmodes
            argument.
        mmodes (array_like):
            Which azimuthal modes to use for the Fourier-Bessel basis
        rho (array_like):
            Radial coordinate on disc.
        phi (array_like):
            Azimuthal coordinate on disc (usually radio astronomer's azimuth angle)
        polarized (bool):
            Whether or not polarized beam inference is being done


    Returns:
        fit_beam (array_like):
            The best-fit Fourier-Bessel coefficients for the input beam.
    """

    spw_axis_present = utils.get_beam_interp_shape(beam)

    bess_matr = get_bess_matr(nmodes, mmodes, rho, phi)
    ncoeff = bess_matr.shape[-1]

    # Before indexing conventions enforced
    rhs_full = beam.interp(
        az_array=phi.flatten(),
        za_array=np.arccos(1 - rho**2).flatten(),
        freq_array=freqs,
    )[0]

    if polarized:
        if spw_axis_present or force_spw_index:
            rhs = rhs_full[:, 0]  # Will have shape Nfeed, Naxes_vec, Nfreq, Nrho * Nphi
        else:
            rhs = rhs_full
    else:
        if spw_axis_present or force_spw_index:
            rhs = rhs_full[
                1:, 0, :1
            ]  # FIXME: analyticbeam gives nans and zeros for all other indices
        else:
            rhs = rhs_full[
                1:, :1
            ]  # FIXME: analyticbeam gives nans and zeros for all other indices

    Npol = 1 + polarized

    # Loop over frequencies
    Nfreqs = len(freqs)
    fit_beam = np.zeros((Nfreqs, ncoeff, Npol, Npol), dtype=complex)

    BT = bess_matr.conj()
    lhs_op = np.tensordot(BT, bess_matr, axes=((0, 1), (0, 1)))
    BT_res = BT.reshape(rho.size, ncoeff)
    for freq_ind in range(Nfreqs):
        for feed_ind in range(Npol):
            for pol_ind in range(Npol):
                rhs_vec = (
                    BT_res * rhs[feed_ind, pol_ind, freq_ind, :, np.newaxis]
                ).sum(axis=0)
                soln = solve(lhs_op, rhs_vec, assume_a="her")
                fit_beam[freq_ind, :, feed_ind, pol_ind] = soln

    # Reshape coefficients array
    fit_beam = np.array(fit_beam)  # Has shape Nfreqs, ncoeffs, Npol, Npol

    return fit_beam


def get_ant_inds(
        ant_samp_ind,
        nants,
    ):
    """
    Get the indices of the antennas that are not being sampled, according to the
    index of the antenna that is being sampled.

    Parameters:
        ant_samp_ind (int):
            The index for the antenna being sampled in the current Gibbs step.
        nants (int):
            The total number of antennas in the problem.

    Returns:
        ant_inds (array_like):
            Boolean array that is length nants and true everywhere but the
            ant_samp_ind.
    """
    ant_inds = np.arange(nants) != ant_samp_ind
    return ant_inds


def select_subarr(
        arr, 
        ant_samp_ind, 
        Nants
    ):
    """
    Select the subarray for anything of the same shape as the visibilities,
    such as the inverse noise variance and its square root.

    Parameters:
        arr (array_like):
            The array from which the subarray will be extracted.
        ant_samp_ind (int):
            The index of the current antenna being sampled.

    Returns:
        subarr (array_like):
            The subarray relevant to the current Gibbs step.
    """
    ant_inds = get_ant_inds(ant_samp_ind, Nants)
    subarr = arr[:, :, :, :, ant_inds, ant_samp_ind]
    return subarr


def get_bess_outer(
        bess_matr
    ):
    """
    Use fancy indexing to get the outer product of the Fourier-Bessel design
    matrix with itself. The conjugation order is swapped relative to the usual
    convention.

    Parameters:
        bess_matr (array_like):
            The Fourier-Bessel design matrix in question.
    Returns:
        bess_matr_outer (array_like):
            The desired outer product.
    """

    return bess_matr[:, :, np.newaxis] * bess_matr.conj()[:, :, :, np.newaxis]


def get_bess_sky_contraction(
    bess_outer,
    ants,
    fluxes,
    ra,
    dec,
    freqs,
    lsts,
    polarized=False,
    precision=1,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
    outer=True,
    ref_beam_response=None,
):
    """
    Contract the outer product of Fourier-Bessel (FB) design matrix with the sky model
    multiplied by the fringe pattern. This is equivalent to doing a visibility 
    simulation for the array in question for each pair of 2d beam basis functions.
    If power=True, instead just do this per beam basis function.

    Parameters:
        bess_outer (array_like):
            Outer product of FB design matrix with itself, unless power=True,
            in which case this should just be the FB design matrix.
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.
        fluxes (array_like):
            2D array with the flux of each source as a function of frequency,
            of shape (NSRCS, NFREQS).
        ra, dec (array_like):
            Arrays of source RA and Dec positions in radians. RA goes from
            [0, 2 pi] and Dec from [-pi, +pi].
        freqs (array_like):
            Frequency channels for the simulation, in Hz.
        lsts (array_like):
            Local sidereal times for the simulation, in radians. Range is
            [0, 2 pi].
        polarized (bool):
            If True, raise a NotImplementedError. Eventually this will use
            polarized beams.
        precision (int):
            Which precision setting to use for :func:`~vis_cpu`. If set to
            ``1``, uses the (``np.float32``, ``np.complex64``) dtypes. If set
            to ``2``, uses the (``np.float64``, ``np.complex128``) dtypes.
        latitude (float):
            The latitude of the center of the array, in radians. The default is
            the HERA latitude = -30.7215 * pi / 180.
        use_feed (str):
            Which feed to use, default is 'x'. TODO: what is 'x' in cardinal directions?
        outer (bool):
            Whether sampling power (baseline) beams rather than antenna beams.
        ref_beam_response (UVBeam):
            A reference beam response evaluated at the ra and dec in question.
    """

    Npol = 2 if polarized else 1
    Nfreqs = len(freqs)
    Ncoeff = bess_outer.shape[-1]
    Ntimes = len(lsts)
    Nants = len(ants)
    contract_shape = [Npol, Npol, Nfreqs, Ntimes, Nants, Nants, Ncoeff]
    if outer:
        contract_shape += [Ncoeff,]

    # tsb,qQftaAs,tsB -> qQftaAbB
    # or #
    # ts,qQftaAs,tsB -> qQftaAB
    bess_sky_contraction = np.zeros(contract_shape, dtype=complex)
    beams = [UniformBeam() for ant_ind in range(len(ants))]

    # This explicit for loop is actually faster than calling opt_einsum!
    for time_ind in range(Ntimes):
        sky_amp_phase = simulate_vis_per_source(
            ants,
            fluxes,
            ra,
            dec,
            freqs,
            lsts[time_ind : time_ind + 1],
            beams=beams,
            polarized=polarized,
            precision=precision,
            latitude=latitude,
            use_feed=use_feed,
        )
        if not polarized:
            sky_amp_phase = sky_amp_phase[np.newaxis, np.newaxis, :]
            if ref_beam_response is not None:
                sky_amp_phase *= ref_beam_response[:, None, None, None, time_ind]
        # Need this conjugation since only lower half of array is filled
        # less memory efficient but this isn't the dominant term
        # makes compute later easier to think about
        sky_amp_phase = sky_amp_phase + sky_amp_phase.swapaxes(4, 5).conj()

        bess_sky_contraction[:, :, :, time_ind] = np.tensordot(
            sky_amp_phase[:, :, :, 0], bess_outer[time_ind], axes=((-1,), (0,))
        )

    return bess_sky_contraction

def get_bess_to_vis_from_contraction(bess_sky_contraction, beam_coeffs, ants,
                                     ant_samp_ind, ref_contraction=False):
    """
    Get a linear operator that maps a particular antenna's beam coefficients to 
    visibilities.

    Parameters:
        bess_sky_contraction (array):
            Output of get_bess_sky_contraction. Quadratic form that maps 
            beam_coeffs to visibilities.
        beam_coeffs (array):
            Complex beam coefficients, shape (Nbasis, Nfreqs, Nants, Npol, Npol)
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.
        ant_samp_ind (int):
            Index of the antenna being sampled.
    """
    Nants = len(ants)
    ant_inds = get_ant_inds(ant_samp_ind, Nants)
    beam_res = (beam_coeffs.transpose((2, 3, 1, 0, 4)))[ant_inds]  # bfApQ -> ApfbQ
    # Experimentation with opt_einsum suggests no clever speedups so just call einsum 
    if ref_contraction:
        einstr = "ApfbQ,qQftAb->pqftA"
    else:
        einstr = "ApfbQ,qQftAbB->pqftAB"
    bess_trans = np.einsum(
        einstr,
        beam_res.conj(),
        bess_sky_contraction[:, :, :, :, ant_inds, ant_samp_ind],
        optimize=True,
    )

    return bess_trans


def get_lin_approx_bess_to_vis(ref_contraction):
    Nants = ref_contraction.shape[-3]
    Npols, Nfreqs, Ntimes, Nants = ref_contraction.shape[1:5]
    assert Npols == 1, "Polarized analysis is not available for this approximation"
    Nbls = (Nants * Nants - 1)//2
    Ncoeff = ref_contraction.shape[-1]
    lin_approx_bess_to_vis = np.zeros([Npols, Npols, Nfreqs, Ntimes, Nbls, Nants, Ncoeff, 2, 2])
    sigz = np.array([[1, 0], 
                     [0, -1]])

    # Ewww, pointer walk
    bl_start = 0
    for ant_ind in range(Nants):
        ant_inds = slice(ant_ind +1, Nants)
        Nants_this_ant = Nants - 1 - ant_ind
        bl_stop = bl_start + Nants_this_ant
        bl_inds = slice(bl_start, bl_stop)
        ref_this_ant = ref_contraction[:, :, :, :, ant_inds, ant_ind]
        ref_this_ant_real = split_real_imag(ref_this_ant)
        lin_approx_bess_to_vis[:, :, :, :, bl_inds, ant_ind] = ref_this_ant_real
        # Talks to conjugate
        lin_approx_bess_to_vis[:, :, :, :, bl_inds, ant_inds] = ref_this_ant_real * sigz
        bl_start += Nants_this_ant
    
    return lin_approx_bess_to_vis


def get_bess_to_vis(
    bess_matr,
    ants,
    fluxes,
    ra,
    dec,
    freqs,
    lsts,
    beam_coeffs,
    ant_samp_ind,
    polarized=False,
    precision=1,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
):
    """
    Slower alternative to get_bess_outer -> get_bess_sky_contraction -> get_bess_to_vis_contraction.
    
    Compute the matrices that act as the quadratic forms by which visibilities
    are made. Calls simulate_vis_per_source to get the Fourier operator, then
    transforms to the Fourier-Bessel basis.

    Parameters:
        bess_matr (array_like, complex):
            Matrix that transforms from Fourier-Bessel coefficients to beam
            value at source position
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.
        fluxes (array_like):
            2D array with the flux of each source as a function of frequency,
            of shape (NSRCS, NFREQS).
        ra, dec (array_like):
            Arrays of source RA and Dec positions in radians. RA goes from
            [0, 2 pi] and Dec from [-pi, +pi].
        freqs (array_like):
            Frequency channels for the simulation, in Hz.
        lsts (array_like):
            Local sidereal times for the simulation, in radians. Range is
            [0, 2 pi].
        beam_coeffs (array_like, real):
            Zernike coefficients for the beam at each freqency and
            antenna. Has shape `(ncoeff, NFREQS, NANTS, Nfeed, Naxes_vec)`.
        ant_samp_ind (int):
            ID of the antenna that is being sampled.
        polarized (bool):
            If True, raise a NotImplementedError. Eventually this will use
            polarized beams.
        precision (int):
            Which precision setting to use for :func:`~vis_cpu`. If set to
            ``1``, uses the (``np.float32``, ``np.complex64``) dtypes. If set
            to ``2``, uses the (``np.float64``, ``np.complex128``) dtypes.
        latitude (float):
            The latitude of the center of the array, in radians. The default is
            the HERA latitude = -30.7215 * pi / 180.
        use_feed (str):
            Which feed to use, default is 'x'. TODO: what is 'x' in cardinal directions?
    Returns:
        bess_trans (array_like):
            Operator that returns visibilities when supplied with
            beam coefficients. Shape (NPOL, NPOL, NFREQS, NTIMES, NANTS, ncoeff)`
            where NPOL=1 if not polarized.
    """

    nants = len(ants)
    ant_inds = get_ant_inds(ant_samp_ind, nants)
    Npol = 2 if polarized else 1

    # Use uniform beams so that we just get the Fourier operator.
    beams = [UniformBeam() for ant_ind in range(len(ants))]
    sky_amp_phase = simulate_vis_per_source(
        ants,
        fluxes,
        ra,
        dec,
        freqs,
        lsts,
        beams=beams,
        polarized=polarized,
        precision=precision,
        latitude=latitude,
        use_feed=use_feed,
        subarr_ant=ant_samp_ind,
    )
    sky_amp_phase = sky_amp_phase[:, :, ant_inds]
    if not polarized:
        sky_amp_phase = sky_amp_phase[np.newaxis, np.newaxis, :]

    # Want to do the contraction tsb,aPQtfs,tsB,aqfBQ->aPqftb
    # Determined optimal contraction order with opt_einsum
    # Implementing steps in faster way using tdot
    # aqfBQ,tsB->aqfQts->Qqftas
    beam_res = (beam_coeffs.transpose((2, 3, 1, 0, 4)))[ant_inds]  # BfaqQ -> aqfBQ
    beam_on_sky = np.tensordot(
        beam_res.conj(), bess_matr.conj(), axes=((3,), (2,))
    ).transpose((3, 1, 2, 4, 0, 5))

    # Qqftas,QPftas->qPftas
    # reassign to save memory
    # Qqftas -> Qq_ftas; QPftas->Q_Pftas; Qq_ftas,Q_Pftas->qPftas
    sky_amp_phase = (beam_on_sky[:, :, np.newaxis] * sky_amp_phase[:, np.newaxis]).sum(
        axis=0
    )

    # qPftas,tsb->qPftab
    bess_trans = np.zeros(
        (Npol, Npol, freqs.size, lsts.size, nants - 1, beam_res.shape[-2]),
        dtype=sky_amp_phase.dtype,
    )

    for time_ind in range(lsts.size):
        bess_trans[:, :, :, time_ind, :, :] = np.tensordot(
            sky_amp_phase[:, :, :, time_ind], bess_matr[time_ind], axes=((-1,), (0,))
        )
    return bess_trans



def get_cov_Qdag_Ninv_Q(inv_noise_var, bess_trans, cov_tuple):
    """
    Construct the nontrivial part of the LHS operator for the Gibbs sampling.

    Parameters:
        inv_noise_var (array_like):
            Inverse variance of same shape as vis. Assumes diagonal covariance
            matrix, which is true in practice.
        cov_tuple (tuple of array):
            Factorized prior covariance matrix in the sense of the tensor product.
            In other words each element is a prior covariance over a given axis
            of the beam coefficients (mode, frequency, complex component), and
            the total covariance is the tensor product of these matrices.
        bess_trans: (array_like): (Complex) matrix that, when applied to a
            vector of Zernike coefficients for one antenna, returns the
            visibilities associated with that antenna.

    Returns:
        cov_Qdag_Ninv_Q (array_like):
            The prior covariance matrix multiplied by the inverse noise covariance
            transformed to the Fourier-Bessel basis.
    """
    freq_matr, comp_matr, bfunc_matr = cov_tuple
    Nfreqs = freq_matr.shape[0]

    # These stay as elementwise multiply since the beam at given times/freqs
    # Should not affect the vis at other times/freqs

    # qpfta,qPftab->qpPftab
    Ninv_Q = (
        inv_noise_var[:, :, np.newaxis, :, :, :, np.newaxis]
        * bess_trans[:, np.newaxis, :, :, :, :, :]
    )

    # qRftab,qpPFtaB->RfbpPFB
    # Actually just want diagonals in frequency but don't need to save memory here
    Qdag_Ninv_Q = np.tensordot(bess_trans.conj(), Ninv_Q, axes=((0, 3, 4), (0, 4, 5)))
    # Get the diagonals RfbpPFB-> fRbpPB
    Qdag_Ninv_Q = Qdag_Ninv_Q[:, range(Nfreqs), :, :, :, range(Nfreqs)]
    # Factor of 2 because 1/2 the variance for each complex component
    Qdag_Ninv_Q = 2 * split_real_imag(Qdag_Ninv_Q, kind="op")  # fRbpPBcC
    Qdag_Ninv_Q = Qdag_Ninv_Q.transpose((1, 2, 3, 4, 5, 7, 6, 0))  # fRbpPBcC->RbpPBCcf

    # c,fF->cfF->fcF
    cov_matr = np.swapaxes(comp_matr[:, np.newaxis, np.newaxis] * freq_matr, 0, 1)
    # fcF,RbpPBCcF->fRbpPBCcF
    cov_Qdag_Ninv_Q = (
        cov_matr[
            :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]
        * Qdag_Ninv_Q[np.newaxis]
    )
    # fRbpPBCcF->fRbpPBcCF, doing this because we should have multiplied the left index, but Qdag_Ninv_Q was symmetric on those two indices, so we multiplied the right for convenience
    cov_Qdag_Ninv_Q = np.swapaxes(cov_Qdag_Ninv_Q, -2, -3)
    # Ab,fRbpPBcCF -> AfRpPBcCF -> fRApPBcCF
    cov_Qdag_Ninv_Q = np.tensordot(bfunc_matr, cov_Qdag_Ninv_Q, axes=((1,), (2,))).transpose(1, 2, 0, 3, 4, 5, 6, 7, 8)

    return cov_Qdag_Ninv_Q


def apply_operator(x, cov_Qdag_Ninv_Q):
    """
    Apply LHS operator to vector of Fourier-Bessel coefficients.

    Parameters:
        x (array_like):
            Complex Fourier-Bessel coefficients, split into real/imag.
        cov_Qdag_Ninv_Q (array_like):
            One summand of LHS matrix applied to x


    Returns:
        Ax (array_like):
            Result of multiplying input vector by the LHS matrix.
    """

    Npol = x.shape[2]
    # Linear so we can split the LHS multiply
    # fRbpPBcCF,BFpPC->fRbpcp->pRfbc->pfbRc
    Ax1 = np.tensordot(cov_Qdag_Ninv_Q, x, axes=((4, 5, 7, 8), (3, 0, 4, 1)))
    Ax1 = (Ax1[:, :, :, range(Npol), :, range(Npol)]).transpose((0, 2, 3, 1, 4))

    # Second term is identity due to preconditioning
    Ax = (Ax1 + x).transpose((1, 2, 0, 3, 4))
    return Ax


def get_std_norm(shape):
    """
    Get an array of complex standard normal samples.

    Parameters:
        shape (array_like): Shape of desired array.

    Returns:
        std_norm (array_like): desired complex samples.
    """
    std_norm = (
        np.random.normal(size=shape) + 1.0j * np.random.normal(size=shape)
    ) / np.sqrt(2)

    return std_norm


def construct_rhs(vis, inv_noise_var, mu, bess_trans, cov_tuple, cho_tuple, flx=True):
    """
    Construct the right hand side of the Gaussian Constrained Realization (GCR)
    equation.

    Parameters:
        vis (array_like):
            Subset of visiblities belonging to
            the antenna for which the GCR is being set up. Has shape
            `(NFREQS, NTIMES, NANTS - 1)`.
        inv_noise_./m,./ vbnvar (array_like):
            Inverse variance of same shape as `vis`. Assumes diagonal
            covariance matrix, which is true in practice.
        mu (array_like):
            Prior mean for the Fourier-Bessel coefficients (pre-calculated).
        cov_tuple (tuple of arr): tensor-factored covariance matrix.
        cho_tuple (tuple of arr): tensor-factored, cholesky decomposed prior
            covariance matrix.
        bess_trans (array_like):
            Operator that maps beam coefficien[l'9ots from one antenna into its
            subset of visibilities.
        flx (bool):
            Whether to use fluctuation terms. Useful for debugging.

    Returns:
        rhs (array_like):
            RHS vector for GCR eqn.
    """

    flx0_shape = mu.shape
    flx1_shape = vis.shape

    if flx:
        flx0 = get_std_norm(flx0_shape)
        flx1 = get_std_norm(flx1_shape)
    else:
        flx0 = np.zeros(flx0_shape)
        flx1 = np.zeros(flx1_shape)

    Ninv_d = inv_noise_var * vis
    Ninv_sqrt_flx1 = np.sqrt(inv_noise_var) * flx1

    # qPftab,qpfta->bfpP
    # qPftab->bPqfta
    bess_trans_use = bess_trans.transpose((5, 1, 0, 2, 3, 4))
    # Weird factors of sqrt(2) etc since we will split these in a sec
    # bPqfta,pqfta->bpPf->bfpP

    Qdag_terms = np.sum(
        bess_trans_use.conj()[:, np.newaxis]
        * (2 * Ninv_d + np.sqrt(2) * Ninv_sqrt_flx1).transpose((1, 0, 2, 3, 4))[
            :, :, np.newaxis
        ],
        axis=(3, 5, 6),
    ).transpose((0, 3, 1, 2))
    Qdag_terms = split_real_imag(Qdag_terms, kind="vec")

    freq_matr, comp_matr, bfunc_matr = cov_tuple
    # c,bfpPc->bfpPc Bb,bfpPc->BfpPc Ff,bfpPc->FbpPc 
    cov_Qdag_terms = np.tensordot(
        bfunc_matr, (comp_matr * Qdag_terms), axes=1
    )
    cov_Qdag_terms = np.tensordot(
        freq_matr, cov_Qdag_terms, axes=((1,), (1,))
    )

    freq_cho, comp_cho, bfunc_cho = cho_tuple
    flx0 = split_real_imag(flx0, kind="vec")
    mu_real_imag = split_real_imag(mu, kind="vec")

    flx0_add = np.tensordot(bfunc_cho, (comp_cho * flx0), axes=1)
    flx0_add = np.tensordot(freq_cho, flx0_add, axes=((1,), (1,)))

    rhs = cov_Qdag_terms + np.swapaxes(mu_real_imag, 0, 1) + flx0_add

    return rhs


def non_norm_gauss(A, sig, x):
    """
    Make a Gaussian function (not a probability density function). Used in
    constructing freq-freq covariance functions.

    Parameters:
        A (float):
            Amplitude of the Gaussian function
        sig (float):
            Width of the Gaussian function
        x (array_like):
            Locations to evaluate the Gaussian function.

    Returns:
        gvals (array_like):
            Values of the Gaussian function at positions given by x
    """

    gvals = A * np.exp(-(x**2) / (2 * sig**2))

    return gvals


def make_prior_cov(
    freqs, 
    std,
    sig_freq, 
    Nbasis,
    constrain_phase=False, 
    constraint=1e-4, 
    ridge=0,
    cov_file=None,
):
    """
    Make a prior covariance for the beam coefficients.

    Parameters:
        freqs (array_like):
            Frequencies over which the covariance matrix is calculated.
        times (array_like):
            Times over which the covariance matrix is calculated.
        ncoeff (int):
            Number of Zernike coefficients in use.
        std (float):
            Square root of the diagonal entries of the matrix.
        sig_freq (float):
            Correlation length in frequency.
        contraint (bool):
            Whether to constrian the phase of the beams or not. Currently just
            constrains them to have only small variations in the imaginary part.
        ridge (float):
            A ridge adjustment for the freq-freq covariance matrix, to make it
            better-conditioned when the correlation length is long.
        cov_file (str):
            Path to file for covariance matrix.
    Returns:
        cov_tuple (array_like):
            Tuple of tensor components of covariance matrix.
    """
    freq_col = non_norm_gauss(1, sig_freq, freqs - freqs[0])
    freq_col[0] += ridge
    freq_matr = toeplitz(freq_col)
    comp_matr = np.ones(2)
    if constrain_phase:  # Make the imaginary variance small compared to the real one
        comp_matr[1] = constraint
    if cov_file is None:
        bfunc_matr = np.eye(Nbasis) * std**2
    else:
        bfunc_matr = np.load(cov_file)

    cov_tuple = (freq_matr, comp_matr, bfunc_matr)

    return cov_tuple

def check_cho(cho, cov):
    prod = cho @ cho.T.conj()
    allclose = np.allclose(prod, cov)
    if not allclose:
        raise LinAlgError(f"Cholesky factorization failed to reproduce covariance")

    return

def do_cov_cho(
        cov_tuple, 
        check_op=False
    ):
    """
    Returns the Cholesky decomposition of a matrix factorable over
    time/frequency/complex component with the same
    shape as the covariance of a given antennas beam coefficients (per time and
    freq).

    Parameters:
        cov_tuple (tuple of arr):
            The factors that make up the operator.

    Returns:
        cho_tuple (array_like):
            The factored Cholesky decomposition.
    """
    freq_matr, comp_matr, bfunc_matr = cov_tuple
    freq_cho = cholesky(freq_matr, lower=True)
    comp_cho = np.sqrt(comp_matr)  # Currently always diagonal
    bfunc_cho = cholesky(bfunc_matr, lower=True)

    cho_tuple = (freq_cho, comp_cho, bfunc_cho)
    if check_op:
        for cho, cov in zip([freq_cho, bfunc_cho], [freq_matr, bfunc_matr]):
            check_cho(cho, cov)

    return cho_tuple


def get_beam_from_FB_coeff(beam_coeffs, za, az, nmodes, mmodes):
    """
    Gets a beam from a list of beam coefficients at desired zenith angle and
    azimuth.

    Parameters:
        beam_coeffs (complex_array): Fourier-Bessel coefficients of a particular
            antenna for a particular frequency.
        za (array): zenith angles in radians
        az (array): azimuths in radians
        nmodes (array_like):
            Radial modes to use for the Bessel basis. Should correspond to mmodes
            argument.
        mmodes (array_like):
            Which azimuthal modes to use for the Fourier-Bessel basis

    Returns:
        beam (array_like): Beam evaluated at a grid of za, az
    """

    rho = np.sqrt(1 - np.cos(za))
    Rho, Az = np.meshgrid(rho, az)
    B = get_bess_matr(nmodes, mmodes, Rho, Az)

    beam = B @ beam_coeffs

    return beam


def plot_FB_beam(
    beam,
    za,
    az,
    vmin=-1,
    vmax=1,
    norm=SymLogNorm,
    linthresh=1e-3,
    cmap="Spectral",
    save=False,
    fn="",
    **kwargs,
):
    """
    Plots a Fourier_Bessel beam at specified zenith angles and azimuths.

    Parameters:
        beam (array_like): Beam evaluated at a grid of za, az
        za (array): zenith angles in radians
        az (array): azimuths in radians
        vmin (float): Minimum value to plot
        vmax (float): Max value to plot
        norm (matplotlib colormap normalization): Which colormap normalization to use.
        linthresh (float): The linear threshold for the SymLogNorm map
        cmap (str): colormap
        kwargs: other keyword arguments for colormap normalization

    Returns:
        None
    """

    Az, Za = np.meshgrid(az, za)

    fig, ax = plt.subplots(ncols=2, subplot_kw={"projection": "polar"}, figsize=(16, 8))
    cax = ax[0].pcolormesh(
        Az,
        Za,
        beam.real,
        norm=norm(vmin=vmin, vmax=vmax, linthresh=linthresh, **kwargs),
        cmap=cmap,
    )
    ax[0].set_title("Real Component")
    ax[1].pcolormesh(
        Az,
        Za,
        beam.imag,
        norm=norm(vmin=vmin, vmax=vmax, linthresh=linthresh, **kwargs),
        cmap=cmap,
    )
    ax[1].set_title("Imaginary Component")
    fig.colorbar(cax, ax=ax.ravel().tolist())
    if save:
        fig.savefig(fn)
        plt.close(fig)

    return


def get_zernike_rad(r, n, m):
    """
    Use hypergeometric representation of radial polynomial to evaluate the
    radial part of the zernike function.

    Parameters:
        r: radial coordinate (arcsin(za))
        n: radial degree
        m: azimuthal degree

    Returns:
        rad: radial polynomial of degree (n,m) evaluated at (theta,r)
    """
    if (n - m) % 2:  # odd difference, return 0
        raise ValueError(
            "Difference between n and m must be even. " f"n={n} and m={m}."
        )
    else:
        nm_diff = (n - m) // 2
        nm_sum = (n + m) // 2
        prefac = (-1) ** nm_diff * comb(nm_sum, m) * r**m
        rad = prefac * hyp2f1(1 + nm_sum, -nm_diff, 1 + m, r**2)

    return rad


def get_zernike_azim(theta, m):
    """
    Get the azimuthal part of a Zernike function

    Parameters:
        theta (array_like): azimuthal coordinate
        m (int): azimuthal degree

    Returns:
        azim (array_like): azimuthal part of Zernike function
    """
    if m >= 0:
        azim = np.cos(m * theta)
    elif m < 0:
        azim = np.sin(np.abs(m) * theta)
    return azim


def get_zernike_matrix(nmax, theta, r):
    ncoeff = (nmax + 1) * (nmax + 2) // 2
    zern_matr = np.zeros((ncoeff,) + theta.shape)

    # Precompute trig functions so that
    # we are not recomputing them over and over for new radial modes
    azim = np.zeros((2 * nmax + 1,) + theta.shape)
    for m in range(-nmax, nmax + 1):
        azim[m] = get_zernike_azim(theta, m)

    # iterate over all modes and assign product of radial/azimuthal basis function
    ind = 0
    for n in range(0, nmax + 1):
        for m in range(-n, n + 1, 2):
            # normalize
            rad = get_zernike_rad(r, n, np.abs(m)) * np.sqrt(2 * n + 2)
            zern_matr[ind] = rad * azim[m] / np.sqrt(np.pi * (1 + (m == 0)))
            ind += 1

    return zern_matr.transpose((1, 2, 0))


def get_pert_beam(
    beam_file,
    outfile,
    cSL=0.2,
    mmax=45,
    nmax=80,
    sqrt=True,
    Nfeeds=2,
    num_modes_comp=32,
    save=False,
    outdir="",
    load=False,
    trans_x=0.,
    trans_y=0.,
    rot=0.,
    stretch_x=1.,
    stretch_y=1.,
    sin_pert_coeffs=np.zeros(8, dtype=float),
):
    """
    Get a perturbed sparse_beam instance.

    Parameters:
        beam_file (str):
            Path to unperturbed beam file.
        outfile (str):
            Where to save the outputs.
        trans_std (float):
            Standard deviation for random tilt of beam, in units of FB radial coordinate.
        rot_std_std (float):
            Standard deviation for random beam rotation, in degrees.
        stretch_std (float):
            Standard deviation for random beam stretching.
        mmax (int):
            The maximum azimuthal mode number to use.
        nmax (int):
            The maximum radial mode number to use in the FB basis.
        sqrt (bool):
            Whether to take the square root of the unperturbed beam before
            fitting. Used for power beams.
        Nfeeds (int):
            Number of feeds. Set to None if using E-field beam, 2 for power beam.
        num_modes_comp (int):
            Does nothing, but will slow down the code if set to a high number.
        save (bool):
            Whether to save the fit coefficients to the perturbed beam.
        outdir (str):
            Path to directory to save output.
        trans_x (float):
            How much to tilt the coordinate system in the x direction 
        trans_y (float):
            How much to tilt the coordinate system in the y direction
        rot (float):
            How many radians to rotate the coordinate sysem
        stretch_x (float):
            How much to stretch the coordinate system in the x direction
        stretch_y (float):
            How much to stretch the coordinate system in the y direction
        sin_pert_coeffs (array):
            Perturbation coefficients for the sidelobes.
    Returns:
        sb (sparse_beam):
            Perturbed sparse_beam instance.
    """

    mmodes = np.arange(-mmax, mmax + 1)
    sb = sparse_beam(
        beam_file,
        nmax,
        mmodes,
        Nfeeds=Nfeeds,
        num_modes_comp=num_modes_comp,
        sqrt=sqrt,
        perturb=True,
        trans_x=trans_x,
        trans_y=trans_y,
        rot=rot,
        stretch_x=stretch_x,
        stretch_y=stretch_y,
        sin_pert_coeffs=sin_pert_coeffs,
        cSL=cSL,
    )


    if load:
        pert_beam = np.load(outfile)
    else:
        Azg, Zag = np.meshgrid(sb.axis1_array, sb.axis2_array)
        pert_beam, _ = sb.interp(az_array=Azg.flatten(), za_array=Zag.flatten())
    fit_coeffs, _ = sb.get_fits(data_array=pert_beam.reshape(sb.data_array.shape))

    if save:
        np.save(outfile, pert_beam)
        np.save(f"{outfile[:outfile.rfind(".npy")]}_fit_coeffs.npy", fit_coeffs)

    return sb


def init_beam_sampler(beams, freqs, beam_mmax, beam_nmax):
    """ """
    beam_nmodes, beam_mmodes = np.meshgrid(
        np.arange(1, beam_nmax + 1), np.arange(-beam_mmax, beam_mmax + 1)
    )
    beam_nmodes = beam_nmodes.flatten()
    beam_mmodes = beam_mmodes.flatten()

    za_fit = np.arange(91) * np.pi / 180
    rho_fit = np.sqrt(1 - np.cos(za_fit)) / args.rho_const
    phi_fit = np.linspace(0, 2 * np.pi, num=360)
    PHI, RHO = np.meshgrid(phi_fit, rho_fit)

    bess_matr_fit = get_bess_matr(beam_nmodes, beam_mmodes, RHO, PHI)

    beam_coeffs_fit = fit_bess_to_beam(
        beams[0], 1e6 * freqs, beam_nmodes, beam_mmodes, RHO, PHI, force_spw_index=True
    )

    print("\tBeam best fit dynamic range:")
    print("\t", np.amax(np.abs(beam_coeffs_fit)), np.amin(np.abs(beam_coeffs_fit)))

    txs, tys, tzs = convert_to_tops(ra, dec, times, array_latitude)

    # area-preserving
    rho = np.sqrt(1 - tzs) / args.rho_const
    phi = np.arctan2(tys, txs)
    bess_matr = hydra.beam_sampler.get_bess_matr(beam_nmodes, beam_mmodes, rho, phi)

    # All the same, so just repeat (for now)
    beam_coeffs = np.array(Nants * [beam_coeffs_fit])
    # Want shape ncoeff, Nfreqs, Nants, Npol, Npol
    beam_coeffs = np.swapaxes(beam_coeffs, 0, 2).astype(complex)
    np.save(os.path.join(output_dir, "best_fit_beam"), beam_coeffs)
    ncoeffs = beam_coeffs.shape[0]

    if PLOTTING:
        plot_beam_cross(beam_coeffs, 0, 0, "_best_fit")

    amp_use = x_soln if SAMPLE_PTSRC_AMPS else ptsrc_amps
    flux_use = get_flux_from_ptsrc_amp(amp_use, freqs, beta_ptsrc)

    # Hardcoded parameters. Make variations smooth in time/freq.
    sig_freq = 0.5 * (freqs[-1] - freqs[0])
    cov_tuple = hydra.beam_sampler.make_prior_cov(
        freqs, times, ncoeffs, args.beam_prior_std, sig_freq, ridge=1e-6
    )
    cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
    cov_tuple_0 = hydra.beam_sampler.make_prior_cov(
        freqs,
        times,
        ncoeffs,
        args.beam_prior_std,
        sig_freq,
        ridge=1e-6,
        constrain_phase=True,
        constraint=1,
    )
    cho_tuple_0 = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)

    # Be lazy and just use the initial guess.
    coeff_mean = beam_coeffs[:, :, 0]
    bess_outer = hydra.beam_sampler.get_bess_outer(bess_matr)

def get_loglike_from_bsc(beam_coeffs, bess_sky_contraction, inv_noise_var, data,
                         ants):
    """
    Calculate the log-likelihood of some beam coefficients.

    Parameters:
        beam_coeffs (array):
            Complex beam coefficients, shape (Nbasis, Nfreqs, Nants, Npol, Npol)
        bess_sky_contraction (array):
            A large quadratic formthat maps beam_coeffs to visibilities. Output
            of get_bess_sky_contraction.
        inv_noise_var (array):
            Inverse noise variance for the visibilities.
        data (array):
            The visibilities.
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.

    Returns:
        loglike (float):
            The natural log of the likelihood for the data/beam coeffs.
    """

    model = np.zeros_like(data)
    Nants = len(ants)
    for ant_ind in range(Nants):
        ant_inds = get_ant_inds(ant_ind, Nants)
        b_to_v = get_bess_to_vis_from_contraction(bess_sky_contraction, 
                                                  beam_coeffs, ants,
                                                  ant_ind)
        #qPftab,bfpq-> ftapP FIXME: NOT SURE IF THIS POL CONVENTION IS RIGHT
        model[:, :, :, :, ant_inds, ant_ind] = np.einsum("qPftab,bfpq->pPfta", 
                                                         b_to_v, 
                                                         beam_coeffs[:, :, ant_ind],
                                                         optimize=True)
        
    res_sq = np.abs(data - model)**2
    # Did not calculate autos above
    res_sq[:, :, :, :, np.arange(Nants), np.arange(Nants)] = 0.
    loglike_exp = -np.sum(res_sq * inv_noise_var)
    loglike_prefac = -(model.size * np.log(2 * np.pi) - np.log(inv_noise_var).sum())

    loglike = loglike_exp + loglike_prefac

    return loglike
        
        
