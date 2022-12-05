import numpy as np
from scipy.linalg import lstsq, toeplitz, cholesky, inv, LinAlgError

from pyuvsim import AnalyticBeam
from pyuvsim.analyticbeam import diameter_to_sigma
from vis_cpu import conversions

from .vis_simulator import simulate_vis_per_source
from . import utils


def split_real_imag(arr, kind):
    """
    Split an array into real and imaginary components in different ways
    depending on whether the array is an operator or a vector.

    Parameters:
        arr (array_like):
            The array for which the components are to be split.
        kind (str):
            The type of array. Either 'op' (for operator) or 'vec' (for vector).

    Returns:
        new_arr (array_like):
            Array that has been split into real and imaginary parts.
    """
    valid_kinds = ['op', 'vec']
    nax = len(arr.shape)
    if kind not in valid_kinds:
        raise ValueError("kind must be 'op' or 'vec' when splitting complex "
                         "arrays into real and imaginary components")
    if kind == 'op':
        new_arr = np.zeros((2, 2) + arr.shape, dtype=float)
        new_arr[0, 0] = arr.real
        new_arr[0, 1] = -arr.imag
        new_arr[1, 0] = arr.imag
        new_arr[1, 1] = arr.real

        # Prepare to put these axes at the end
        axes = list(range(2, nax + 2)) + [0, 1]

    elif kind == 'vec':
        new_arr = np.zeros((2,) + arr.shape, dtype=float)
        new_arr[0], new_arr[1] = (arr.real, arr.imag)

        # Prepare to put this axis at the end
        axes = list(range(1, nax + 1)) + [0, ]

    # Rearrange axes (they are always expected at the end)
    new_arr = np.transpose(new_arr, axes=axes)
    return new_arr


def reshape_data_arr(arr, Nfreqs, Ntimes, Nants):
    """
    Reshape a data-shaped array into something more convenient for beam calculation.
    Makes a copy of the data twice as large as the data.

    Parameters:
        arr (array_like, complex):
            Array to be reshaped.
        Nfreqs (int):
            Number of frequencies in the data.
        Ntimes (int):
            Number of times in the data.
        Nants (int):
            Number of antennas in the data.

    Returns:
        arr_beam (array_like, complex):
            The reshaped array
    """

    arr_trans = np.transpose(arr, (1, 2, 0))
    arr_beam = np.zeros([Nfreqs, Ntimes, Nants, Nants], dtype=arr_trans.dtype)
    for freq_ind in range(Nfreqs):
        for time_ind in range(Ntimes):
                triu_inds = np.triu_indices(Nants, k=1)
                arr_beam[freq_ind, time_ind, triu_inds[0], triu_inds[1]] = arr_trans[freq_ind, time_ind]

    return arr_beam


def construct_zernike_matrix(nmax, txs, tys):
    """
    Make the matrix that transforms from the Zernike basis to direction cosines.

    Parameters:
        nmax (int):
            Maximum radial degree to use for the Zernike polynomial basis.
            Can range from 0 to 10.
        txs (array_like):
            East-West direction cosine.
        tys (array_like):
            North-South direction cosine.

    Returns:
        Zmatr (array_like):
            Zernike transformation matrix.
    """
    # Just get the polynomials at the specified ls and ms
    Zdict = zernike(np.ones(66), np.array(txs), np.array(tys))

    # Analytic formula  from inspecting zernike function
    ncoeff = (nmax + 1) * (nmax + 2) // 2

    # The order of this reshape depends on what order the convert_to_tops spits
    # out the answer in.
    Zmatr = np.array(list(Zdict.values())[:ncoeff])
    Zmatr = np.transpose(Zmatr, axes=(1, 2, 0))
    return Zmatr


def fit_zernike_to_beam(beam, freqs, Zmatr, txs, tys):
    """
    Get the best fit Zernike coefficients for a beam based on its value at a
    a set of source positions at certain sidereal times, from a fixed latitude,
    at a set of frequencies. A least-squares algorithm is used to perform the
    fits.

    Parameters:
        beam (pyuvsim.Beam):
            The beam being fitted.
        nmax (int):
            Maximum radial mode of the Zernike fit.
        lsts (array_like):
            Sidereal times of the observation.
        latitude (float):
            Latitude of observing, in radians.
        freqs (array_like):
            Frequencies of obseration.

    Returns:
        fit_beam (array_like):
            The best-fit Zernike coefficients for the input beam.
    """
    Ntimes, Nsource, ncoeff = Zmatr.shape
    Nfreqs = len(freqs)

    # Diagonal so just iterate over times
    # There is a faster way to do this using scipy.sparse and encoding Z
    # as block-diagonal in time and frequency
    fit_beam = []
    rhss = []
    for tind, (tx, ty) in enumerate(zip(txs, tys)):
        az, za = conversions.enu_to_az_za(enu_e=tx,
                                          enu_n=ty,
                                          orientation="uvbeam")
        rhs = beam.interp(az, za, freqs)[0][1, 0, 0]
        rhs = np.swapaxes(rhs, 0, 1) # Swap source/freq axis
        rhss.append(rhs)

    rhss = np.reshape(rhss, (Ntimes * Nsource, Nfreqs))

    # Loop over frequencies
    for freq_ind in range(Nfreqs):
        fit_beam_f = lstsq(Zmatr.reshape(Ntimes*Nsource, ncoeff), rhss[:, freq_ind])[0]
        fit_beam.append(fit_beam_f)

    # Reshape coefficients array
    fit_beam = np.array(fit_beam) # Has shape Nfreqs, ncoeffs

    return fit_beam


def get_ant_inds(ant_samp_ind, nants):
    """
    Get the indices of the antennas that are not being sampled, according to the
    index of the antenna that is being sampled.

    Parameters:
        ant_samp_ind (int):
            The index for the antenna being sampled in the current Gibbs step.
        nants (int):
            The total number of antennas in the problem.

    Returns:
        ant_inds (tuple):
            A tuple with one entry, which is itself a 1D array of indices
            corresponding to the antennas that are being conditioned on in the
            current Gibbs step (an output of np.where on a 1D array).
    """
    ant_inds = np.arange(nants) != ant_samp_ind
    return ant_inds


def select_subarr(arr, ant_samp_ind, Nants):
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
    subarr = arr[:, :, ant_inds, ant_samp_ind]
    return subarr


def construct_zernike_to_vis(Zmatr, ants, fluxes, ra, dec, freqs, lsts,
                             beam_coeffs, ant_samp_ind, nants,
                             polarized=False, precision=1,
                             latitude=-30.7215 * np.pi / 180.0, use_feed="x",
                             multiprocess=True, low_mem=False):
    """
    Compute the matrices that act as the quadratic forms by which visibilities
    are made. Calls simulate_vis_per_source to get the Fourier operator, then
    transforms to the Zernike basis.

    Parameters:
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
            antenna. Has shape `(ncoeff, NFREQS, NANTS)`.
        ant_samp_ind (int):
            ID of the antenna that is being sampled.
        nants (int):
            Number of antennas in use.
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
        multiprocess (bool): Whether to use multiprocessing to speed up the
            calculation


    Returns:
        zern_trans (array_like):
            Operator that returns visibilities when supplied with
            beam coefficients. Shape (NFREQS, NTIMES, NANTS, ncoeff)`.
    """
    if polarized:
        raise NotImplementedError("Polarized beams are not available yet.")

    # Use uniform beams so that we just get the Fourier operator.
    beams = [AnalyticBeam("uniform") for ant_ind in range(len(ants))]
    sky_amp_phase = simulate_vis_per_source(ants, fluxes, ra, dec, freqs, lsts,
                                            beams=beams, polarized=polarized,
                                            precision=precision,
                                            latitude=latitude,
                                            use_feed=use_feed,
                                            multiprocess=multiprocess,
                                            subarr_ant=ant_samp_ind)


    # Want to do the contraction zfa,tsz,ftas,tsZ -> ftaZ
    # Determined optimal contraction order with opt_einsum
    # Implementing steps in faster way using tdot

    ant_inds = get_ant_inds(ant_samp_ind, nants)
    # 'zfa,tsz->ftas'
    beam_res = np.swapaxes(beam_coeffs.conj()[:, :, ant_inds], 0, -1) # afz
    # one-liner to save memory
    # afz,tsz->afts->ftas ftas,ftas->ftas
    sky_amp_phase *= np.tensordot(beam_res, Zmatr.conj(), axes=((-1,), (-1,))).transpose(beam_on_sky, axes=(1, 2, 0, 3))

    zern_trans = np.zeros((freqs.size, lsts.size, nants - 1, beam_res.shape[-1]),
                          dtype=sky_amp_phase.dtype)
    for time_ind in range(lsts.size):
        zern_trans[:, time_ind, :, :] = np.tensordot(sky_amp_phase[:, time_ind],
                                                     np.swapaxes(Zmatr, -1, -2)[time_ind],
                                                     axes=((-1, ), (-1, )))
    return zern_trans


def get_cov_Tdag_Ninv_T(inv_noise_var, zern_trans, cov_tuple):
    """
    Construct the LHS operator for the Gibbs sampling.

    Parameters:
        inv_noise_var (array_like):
            Inverse variance of same shape as vis. Assumes diagonal covariance
            matrix, which is true in practice.
        cov_tuple (tuple of array):
            Factorized prior covariance matrix (in the sense of the outer product).
        zern_trans: (array_like): (Complex) matrix that, when applied to a
            vector of Zernike coefficients for one antenna, returns the
            visibilities associated with that antenna.
    """
    freq_matr, comp_matr = cov_tuple
    Nfreqs = freq_matr.shape[0]

    # These stay as elementwise multiply since the beam at given times/freqs
    # Should not affect the vis at other times/freqs
    # ftaZ->Zfta fta,Zfta->Zfta
    zern_trans_use = zern_trans.transpose(axes=(3, 0, 1, 2))
    Ninv_T = inv_noise_var * zern_trans_use
    # zfta,ZFta->zfZF
    # Actually just want diagonals but don't need to save memory here
    Tdag_Ninv_T = np.tensordot(zern_trans_use.conj(), Ninv_T,
                              axes=((-1, -2), (-1, -2)))
    # Get the diagonals zfZF -> fzZ
    Tdag_Ninv_T = Tdag_Ninv_T[:, range(Nfreqs), :, range(Nfreqs)]
    # Factor of 2 because 1/2 the variance for each complex component
    Tdag_Ninv_T = 2*split_real_imag(Tdag_Ninv_T, kind='op')

    # cfF,FzZcC->fFzZcC
    # c,fF->cfF
    cov_matr = comp_matr[:, np.newaxis, np.newaxis] * freq_matr
    # fcF,CzZcF->fCzZcF
    cov_Tdag_Ninv_T = np.swapaxes(cov_matr, 0, 1)[:, np.newaxis, np.newaxis] * np.swapaxes(Tdag_Ninv_T, 0, -1)
    cov_Tdag_Ninv_T = np.transpose(cov_Tdag_Ninv_T, axes=(0, 2, 4, 5, 3, 1))


    return cov_Tdag_Ninv_T


def apply_operator(x, cov_Tdag_Ninv_T):
    """
    Apply LHS operator to vector of Zernike coefficients.

    Parameters:
        x (array_like):
            Complex zernike coefficients, split into real/imag.
        cov_Tdag_Ninv_T (array_like):
            One summand of LHS matrix applied to x


    Returns:
        Ax (array_like):
            Result of multiplying input vector by the LHS matrix.
    """


    # Linear so we can split the LHS multiply
    # fzcFZC,FZC -> fzc
    Ax1 = np.tensordot(cov_Tdag_Ninv_T, x, axes=((-1, -2, -3), (-1, -2, -3)))

    # Second term is identity due to preconditioning
    Ax = Ax1 + x
    return Ax


def construct_rhs(vis, inv_noise_var, inv_noise_var_sqrt, mu, cov_Tdag,
                  cho_tuple, flx=True):
    """
    Construct the right hand side of the Gaussian Constrained Realization (GCR)
    equation.

    Parameters:
        vis (array_like, real):
            Subset of visiblities belonging to the antenna for which the GCR is
            being set up. Split into real and imaginary components. Has shape
            `(NFREQS, NTIMES, NANTS - 1, 2)`.
        inv_noise_var (array_like):
            Inverse variance of same shape as `vis`. Assumes diagonal
            covariance matrix, which is true in practice.
        inv_noise_var_sqrt (array_like):
            Inverse variance of same shape as vis. Assumes diagonal covariance
            matrix, which is true in practice.
        mu (array_like):
            (Complex) prior mean for the Zernike coefficients (pre-calculated).
        cho_tuple (tuple of arr): tensor-factored, cholesky decomposed prior
            covariance matrix.
        cov_Tdag (array_like):
            Matrix product of prior covariance and zernike_to_vis conjugate transpose.
        flx (bool):
            Whether to use fluctuation terms. Useful for debugging.

    Returns:
        rhs (array_like):
            RHS vector.
    """

    flx0_shape = mu.shape
    flx1_shape = vis.shape

    if flx:
        flx0 = np.random.normal(size=flx0_shape) / np.sqrt(2)
        flx1 = np.random.normal(size=flx1_shape) / np.sqrt(2)

    else:
        flx0 = np.zeros(flx0_shape)
        flx1 = np.zeros(flx1_shape)


    Ninv_d = inv_noise_var * vis
    N_inv_sqrt_flx1 = inv_noise_var_sqrt * flx1
    cov_Tdag_terms = np.einsum(
                               'fazcFTC,FTaC -> fzc',
                               cov_Tdag,
                               Ninv_d + N_inv_sqrt_flx1,
                               optimize=True
                               )


    freq_cho, comp_cho = cho_tuple
    flx0_add = np.einsum(
                         'fF,c,zFc->fzc',
                         freq_cho,
                         comp_cho,
                         flx0,
                         optimize=True
                         )
    b = cov_Tdag_terms + flx0_add + np.swapaxes(mu, 0, 1)

    return b


def non_norm_gauss(A, sig, x):
    return A * np.exp(-x**2 / (2 * sig**2))

def make_prior_cov(freqs, times, ncoeff, std, sig_freq,
                   constrain_phase=False, constraint=1e-4, ridge=0):
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
        contrain_phase (bool):
            Whether to constrian the phase of the beams or not. Currently just
            constrains them to have only small variations in the imaginary part.

    Returns:
        cov_tuple (array_like):
            Tuple of tensor components of covariance matrix.
    """
    freq_col = non_norm_gauss(std**2, sig_freq, freqs - freqs[0])
    freq_col[0] += ridge
    freq_matr = toeplitz(freq_col)
    comp_matr = np.ones(2)
    if constrain_phase: # Make the imaginary variance small compared to the real one
        comp_matr[1] = constraint


    cov_tuple = (freq_matr, comp_matr)

    return cov_tuple

def do_cov_cho(cov_tuple, check_op=False):
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
    freq_matr, comp_matr = cov_tuple
    freq_cho = cholesky(freq_matr, lower=True)
    comp_cho = np.sqrt(comp_matr) # Currently always diagonal

    cho_tuple = (freq_cho, comp_cho)
    if check_op:
        prod = freq_cho @ freq_cho.T.conj()
        allclose = np.allclose(prod, freq_matr)
        print(f"Successful cholesky factorization of beam for frequency covariance: "
              f"{allclose}")
        if not allclose:
            raise LinAlgError(f"Cholesky factorization failed for frequency covariance")

    return cho_tuple

def get_chi2(Mjk, beam_coeffs, data, inv_noise_var):
    """
    Get the chi-square for a given set of beam coefficients and data.

    Parameters:
        Mjk (array_like, complex): Quadratic form that maps beam coefficients to
            visibilities. Has shape (Ncoeff, Nfreqs, Ntimes, Nants, Nants, Ncoeffs)
        beam_coeffs (array_like, complex): Has shape (Ncoeff, Nfreqs, Nants)
        data (array_like, complex): Has shape (Nfreqs, Ntimes, Nants, Nants)
    """
    model = np.einsum('zfa,zftaAZ,ZfA->ftaA', beam_coeffs.conj(),
                      Mjk, beam_coeffs, optimize=True)
    Nants = data.shape[-1]
    # zero out the diagonals
    model[:, :, range(Nants), range(Nants)] = 0
    dmm = data - model
    chi2 = np.sum(dmm.conj() * dmm * inv_noise_var)
    return(dmm, chi2)


def zernike(coeffs, x, y):
    """
    Zernike polynomials (up to degree 66) on the unit disc.
    This code was adapted from:
    https://gitlab.nrao.edu/pjaganna/zcpb/-/blob/master/zernikeAperture.py

    Parameters:
        coeffs (array_like):
            Array of real coefficients of the Zernike polynomials, from 0..66.
        x, y (array_like):
            Points on the unit disc.

    Returns:
        zernike (array_like):
            Values of the Zernike polynomial at the input x,y points.
    """
    # Coefficients
    assert len(coeffs) <= 66, "Max. number of coeffs is 66."
    c = np.zeros(66)
    c[: len(coeffs)] += coeffs

    # Precompute powers of x and y
    x2, x3, x4, x5, x6, x7, x8, x9, x10 = tuple(x ** idx for idx in range(2, 11))
    y2, y3, y4, y5, y6, y7, y8, y9, y10 = tuple(y ** idx for idx in range(2, 11))

    # Setting the equations for the Zernike polynomials
    # r = np.sqrt(powl(x,2) + powl(y,2))
    Z = {
        1: c[0] * np.ones_like(x),  # m = 0    n = 0
        2: c[1] * x,  # m = -1   n = 1
        3: c[2] * y,  # m = 1    n = 1
        4: c[3] * 2 * x * y,  # m = -2   n = 2
        5: c[4] * (2 * x2 + 2 * y2 - 1),  # m = 0  n = 2
        6: c[5] * (-1 * x2 + y2),  # m = 2  n = 2
        7: c[6] * (-1 * x3 + 3 * x * y2),  # m = -3     n = 3
        8: c[7] * (-2 * x + 3 * (x3) + 3 * x * (y2)),  # m = -1   n = 3
        9: c[8] * (-2 * y + 3 * y3 + 3 * (x2) * y),  # m = 1    n = 3
        10: c[9] * (y3 - 3 * (x2) * y),  # m = 3 n =3
        11: c[10] * (-4 * (x3) * y + 4 * x * (y3)),  # m = -4    n = 4
        12: c[11] * (-6 * x * y + 8 * (x3) * y + 8 * x * (y3)),  # m = -2   n = 4
        13: c[12]
        * (
            1 - 6 * x2 - 6 * y2 + 6 * x4 + 12 * (x2) * (y2) + 6 * y4
        ),  # m = 0  n = 4
        14: c[13] * (3 * x2 - 3 * y2 - 4 * x4 + 4 * y4),  # m = 2    n = 4
        15: c[14] * (x4 - 6 * (x2) * (y2) + y4),  # m = 4   n = 4
        16: c[15] * (x5 - 10 * (x3) * y2 + 5 * x * (y4)),  # m = -5   n = 5
        17: c[16]
        * (
            4 * x3 - 12 * x * (y2) - 5 * x5 + 10 * (x3) * (y2) + 15 * x * y4
        ),  # m =-3     n = 5
        18: c[17]
        * (
            3 * x
            - 12 * x3
            - 12 * x * (y2)
            + 10 * x5
            + 20 * (x3) * (y2)
            + 10 * x * (y4)
        ),  # m= -1  n = 5
        19: c[18]
        * (
            3 * y
            - 12 * y3
            - 12 * y * (x2)
            + 10 * y5
            + 20 * (y3) * (x2)
            + 10 * y * (x4)
        ),  # m = 1  n = 5
        20: c[19]
        * (
            -4 * y3 + 12 * y * (x2) + 5 * y5 - 10 * (y3) * (x2) - 15 * y * x4
        ),  # m = 3   n = 5
        21: c[20] * (y5 - 10 * (y3) * x2 + 5 * y * (x4)),  # m = 5 n = 5
        22: c[21]
        * (6 * (x5) * y - 20 * (x3) * (y3) + 6 * x * (y5)),  # m = -6 n = 6
        23: c[22]
        * (
            20 * (x3) * y - 20 * x * (y3) - 24 * (x5) * y + 24 * x * (y5)
        ),  # m = -4   n = 6
        24: c[23]
        * (
            12 * x * y
            + 40 * (x3) * y
            - 40 * x * (y3)
            + 30 * (x5) * y
            + 60 * (x3) * (y3)
            - 30 * x * (y5)
        ),  # m = -2   n = 6
        25: c[24]
        * (
            -1
            + 12 * (x2)
            + 12 * (y2)
            - 30 * (x4)
            - 60 * (x2) * (y2)
            - 30 * (y4)
            + 20 * (x6)
            + 60 * (x4) * y2
            + 60 * (x2) * (y4)
            + 20 * (y6)
        ),  # m = 0   n = 6
        26: c[25]
        * (
            -6 * (x2)
            + 6 * (y2)
            + 20 * (x4)
            - 20 * (y4)
            - 15 * (x6)
            - 15 * (x4) * (y2)
            + 15 * (x2) * (y4)
            + 15 * (y6)
        ),  # m = 2   n = 6
        27: c[26]
        * (
            -5 * (x4)
            + 30 * (x2) * (y2)
            - 5 * (y4)
            + 6 * (x6)
            - 30 * (x4) * y2
            - 30 * (x2) * (y4)
            + 6 * (y6)
        ),  # m = 4    n = 6
        28: c[27]
        * (-1 * (x6) + 15 * (x4) * (y2) - 15 * (x2) * (y4) + y6),  # m = 6   n = 6
        29: c[28]
        * (
            -1 * (x7) + 21 * (x5) * (y2) - 35 * (x3) * (y4) + 7 * x * (y6)
        ),  # m = -7    n = 7
        30: c[29]
        * (
            -6 * (x5)
            + 60 * (x3) * (y2)
            - 30 * x * (y4)
            + 7 * x7
            - 63 * (x5) * (y2)
            - 35 * (x3) * (y4)
            + 35 * x * (y6)
        ),  # m = -5    n = 7
        31: c[30]
        * (
            -10 * (x3)
            + 30 * x * (y2)
            + 30 * x5
            - 60 * (x3) * (y2)
            - 90 * x * (y4)
            - 21 * x7
            + 21 * (x5) * (y2)
            + 105 * (x3) * (y4)
            + 63 * x * (y6)
        ),  # m =-3       n = 7
        32: c[31]
        * (
            -4 * x
            + 30 * x3
            + 30 * x * (y2)
            - 60 * (x5)
            - 120 * (x3) * (y2)
            - 60 * x * (y4)
            + 35 * x7
            + 105 * (x5) * (y2)
            + 105 * (x3) * (y4)
            + 35 * x * (y6)
        ),  # m = -1  n = 7
        33: c[32]
        * (
            -4 * y
            + 30 * y3
            + 30 * y * (x2)
            - 60 * (y5)
            - 120 * (y3) * (x2)
            - 60 * y * (x4)
            + 35 * y7
            + 105 * (y5) * (x2)
            + 105 * (y3) * (x4)
            + 35 * y * (x6)
        ),  # m = 1   n = 7
        34: c[33]
        * (
            10 * (y3)
            - 30 * y * (x2)
            - 30 * y5
            + 60 * (y3) * (x2)
            + 90 * y * (x4)
            + 21 * y7
            - 21 * (y5) * (x2)
            - 105 * (y3) * (x4)
            - 63 * y * (x6)
        ),  # m =3     n = 7
        35: c[34]
        * (
            -6 * (y5)
            + 60 * (y3) * (x2)
            - 30 * y * (x4)
            + 7 * y7
            - 63 * (y5) * (x2)
            - 35 * (y3) * (x4)
            + 35 * y * (x6)
        ),  # m = 5  n = 7
        36: c[35]
        * (y7 - 21 * (y5) * (x2) + 35 * (y3) * (x4) - 7 * y * (x6)),  # m = 7  n = 7
        37: c[36]
        * (
            -8 * (x7) * y + 56 * (x5) * (y3) - 56 * (x3) * (y5) + 8 * x * (y7)
        ),  # m = -8  n = 8
        38: c[37]
        * (
            -42 * (x5) * y
            + 140 * (x3) * (y3)
            - 42 * x * (y5)
            + 48 * (x7) * y
            - 112 * (x5) * (y3)
            - 112 * (x3) * (y5)
            + 48 * x * (y7)
        ),  # m = -6  n = 8
        39: c[38]
        * (
            -60 * (x3) * y
            + 60 * x * (y3)
            + 168 * (x5) * y
            - 168 * x * (y5)
            - 112 * (x7) * y
            - 112 * (x5) * (y3)
            + 112 * (x3) * (y5)
            + 112 * x * (y7)
        ),  # m = -4   n = 8
        40: c[39]
        * (
            -20 * x * y
            + 120 * (x3) * y
            + 120 * x * (y3)
            - 210 * (x5) * y
            - 420 * (x3) * (y3)
            - 210 * x * (y5)
            - 112 * (x7) * y
            + 336 * (x5) * (y3)
            + 336 * (x3) * (y5)
            + 112 * x * (y7)
        ),  # m = -2   n = 8
        41: c[40]
        * (
            1
            - 20 * x2
            - 20 * y2
            + 90 * x4
            + 180 * (x2) * (y2)
            + 90 * y4
            - 140 * x6
            - 420 * (x4) * (y2)
            - 420 * (x2) * (y4)
            - 140 * (y6)
            + 70 * x8
            + 280 * (x6) * (y2)
            + 420 * (x4) * (y4)
            + 280 * (x2) * (y6)
            + 70 * y8
        ),  # m = 0    n = 8
        42: c[41]
        * (
            10 * x2
            - 10 * y2
            - 60 * x4
            + 105 * (x4) * (y2)
            - 105 * (x2) * (y4)
            + 60 * y4
            + 105 * x6
            - 105 * y6
            - 56 * x8
            - 112 * (x6) * (y2)
            + 112 * (x2) * (y6)
            + 56 * y8
        ),  # m = 2  n = 8
        43: c[42]
        * (
            15 * x4
            - 90 * (x2) * (y2)
            + 15 * y4
            - 42 * x6
            + 210 * (x4) * (y2)
            + 210 * (x2) * (y4)
            - 42 * y6
            + 28 * x8
            - 112 * (x6) * (y2)
            - 280 * (x4) * (y4)
            - 112 * (x2) * (y6)
            + 28 * y8
        ),  # m = 4     n = 8
        44: c[43]
        * (
            7 * x6
            - 105 * (x4) * (y2)
            + 105 * (x2) * (y4)
            - 7 * y6
            - 8 * x8
            + 112 * (x6) * (y2)
            - 112 * (x2) * (y6)
            + 8 * y8
        ),  # m = 6    n = 8
        45: c[44]
        * (
            x8 - 28 * (x6) * (y2) + 70 * (x4) * (y4) - 28 * (x2) * (y6) + y8
        ),  # m = 8     n = 9
        46: c[45]
        * (
            x9
            - 36 * (x7) * (y2)
            + 126 * (x5) * (y4)
            - 84 * (x3) * (y6)
            + 9 * x * (y8)
        ),  # m = -9     n = 9
        47: c[46]
        * (
            8 * x7
            - 168 * (x5) * (y2)
            + 280 * (x3) * (y4)
            - 56 * x * (y6)
            - 9 * x9
            + 180 * (x7) * (y2)
            - 126 * (x5) * (y4)
            - 252 * (x3) * (y6)
            + 63 * x * (y8)
        ),  # m = -7    n = 9
        48: c[47]
        * (
            21 * x5
            - 210 * (x3) * (y2)
            + 105 * x * (y4)
            - 56 * x7
            + 504 * (x5) * (y2)
            + 280 * (x3) * (y4)
            - 280 * x * (y6)
            + 36 * x9
            - 288 * (x7) * (y2)
            - 504 * (x5) * (y4)
            + 180 * x * (y8)
        ),  # m = -5    n = 9
        49: c[48]
        * (
            20 * x3
            - 60 * x * (y2)
            - 105 * x5
            + 210 * (x3) * (y2)
            + 315 * x * (y4)
            + 168 * x7
            - 168 * (x5) * (y2)
            - 840 * (x3) * (y4)
            - 504 * x * (y6)
            - 84 * x9
            + 504 * (x5) * (y4)
            + 672 * (x3) * (y6)
            + 252 * x * (y8)
        ),  # m = -3  n = 9
        50: c[49]
        * (
            5 * x
            - 60 * x3
            - 60 * x * (y2)
            + 210 * x5
            + 420 * (x3) * (y2)
            + 210 * x * (y4)
            - 280 * x7
            - 840 * (x5) * (y2)
            - 840 * (x3) * (y4)
            - 280 * x * (y6)
            + 126 * x9
            + 504 * (x7) * (y2)
            + 756 * (x5) * (y4)
            + 504 * (x3) * (y6)
            + 126 * x * (y8)
        ),  # m = -1   n = 9
        51: c[50]
        * (
            5 * y
            - 60 * y3
            - 60 * y * (x2)
            + 210 * y5
            + 420 * (y3) * (x2)
            + 210 * y * (x4)
            - 280 * y7
            - 840 * (y5) * (x2)
            - 840 * (y3) * (x4)
            - 280 * y * (x6)
            + 126 * y9
            + 504 * (y7) * (x2)
            + 756 * (y5) * (x4)
            + 504 * (y3) * (x6)
            + 126 * y * (x8)
        ),  # m = 1   n = 9
        52: c[51]
        * (
            -20 * y3
            + 60 * y * (x2)
            + 105 * y5
            - 210 * (y3) * (x2)
            - 315 * y * (x4)
            - 168 * y7
            + 168 * (y5) * (x2)
            + 840 * (y3) * (x4)
            + 504 * y * (x6)
            + 84 * y9
            - 504 * (y5) * (x4)
            - 672 * (y3) * (x6)
            - 252 * y * (x8)
        ),  # m = 3  n = 9
        53: c[52]
        * (
            21 * y5
            - 210 * (y3) * (x2)
            + 105 * y * (x4)
            - 56 * y7
            + 504 * (y5) * (x2)
            + 280 * (y3) * (x4)
            - 280 * y * (x6)
            + 36 * y9
            - 288 * (y7) * (x2)
            - 504 * (y5) * (x4)
            + 180 * y * (x8)
        ),  # m = 5     n = 9
        54: c[53]
        * (
            -8 * y7
            + 168 * (y5) * (x2)
            - 280 * (y3) * (x4)
            + 56 * y * (x6)
            + 9 * y9
            - 180 * (y7) * (x2)
            + 126 * (y5) * (x4)
            - 252 * (y3) * (x6)
            - 63 * y * (x8)
        ),  # m = 7     n = 9
        55: c[54]
        * (
            y9
            - 36 * (y7) * (x2)
            + 126 * (y5) * (x4)
            - 84 * (y3) * (x6)
            + 9 * y * (x8)
        ),  # m = 9       n = 9
        56: c[55]
        * (
            10 * (x9) * y
            - 120 * (x7) * (y3)
            + 252 * (x5) * (y5)
            - 120 * (x3) * (y7)
            + 10 * x * (y9)
        ),  # m = -10   n = 10
        57: c[56]
        * (
            72 * (x7) * y
            - 504 * (x5) * (y3)
            + 504 * (x3) * (y5)
            - 72 * x * (y7)
            - 80 * (x9) * y
            + 480 * (x7) * (y3)
            - 480 * (x3) * (y7)
            + 80 * x * (y9)
        ),  # m = -8    n = 10
        58: c[57]
        * (
            270 * (x9) * y
            - 360 * (x7) * (y3)
            - 1260 * (x5) * (y5)
            - 360 * (x3) * (y7)
            + 270 * x * (y9)
            - 432 * (x7) * y
            + 1008 * (x5) * (y3)
            + 1008 * (x3) * (y5)
            - 432 * x * (y7)
            + 168 * (x5) * y
            - 560 * (x3) * (y3)
            + 168 * x * (y5)
        ),  # m = -6   n = 10
        59: c[58]
        * (
            140 * (x3) * y
            - 140 * x * (y3)
            - 672 * (x5) * y
            + 672 * x * (y5)
            + 1008 * (x7) * y
            + 1008 * (x5) * (y3)
            - 1008 * (x3) * (y5)
            - 1008 * x * (y7)
            - 480 * (x9) * y
            - 960 * (x7) * (y3)
            + 960 * (x3) * (y7)
            + 480 * x * (y9)
        ),  # m = -4   n = 10
        60: c[59]
        * (
            30 * x * y
            - 280 * (x3) * y
            - 280 * x * (y3)
            + 840 * (x5) * y
            + 1680 * (x3) * (y3)
            + 840 * x * (y5)
            - 1008 * (x7) * y
            - 3024 * (x5) * (y3)
            - 3024 * (x3) * (y5)
            - 1008 * x * (y7)
            + 420 * (x9) * y
            + 1680 * (x7) * (y3)
            + 2520 * (x5) * (y5)
            + 1680 * (x3) * (y7)
            + 420 * x * (y9)
        ),  # m = -2   n = 10
        61: c[60]
        * (
            -1
            + 30 * x2
            + 30 * y2
            - 210 * x4
            - 420 * (x2) * (y2)
            - 210 * y4
            + 560 * x6
            + 1680 * (x4) * (y2)
            + 1680 * (x2) * (y4)
            + 560 * y6
            - 630 * x8
            - 2520 * (x6) * (y2)
            - 3780 * (x4) * (y4)
            - 2520 * (x2) * (y6)
            - 630 * y8
            + 252 * x10
            + 1260 * (x8) * (y2)
            + 2520 * (x6) * (y4)
            + 2520 * (x4) * (y6)
            + 1260 * (x2) * (y8)
            + 252 * y10
        ),  # m = 0    n = 10
        62: c[61]
        * (
            -15 * x2
            + 15 * y2
            + 140 * x4
            - 140 * y4
            - 420 * x6
            - 420 * (x4) * (y2)
            + 420 * (x2) * (y4)
            + 420 * y6
            + 504 * x8
            + 1008 * (x6) * (y2)
            - 1008 * (x2) * (y6)
            - 504 * y8
            - 210 * x10
            - 630 * (x8) * (y2)
            - 420 * (x6) * (y4)
            + 420 * (x4) * (y6)
            + 630 * (x2) * (y8)
            + 210 * y10
        ),  # m = 2  n = 10
        63: c[62]
        * (
            -35 * x4
            + 210 * (x2) * (y2)
            - 35 * y4
            + 168 * x6
            - 840 * (x4) * (y2)
            - 840 * (x2) * (y4)
            + 168 * y6
            - 252 * x8
            + 1008 * (x6) * (y2)
            + 2520 * (x4) * (y4)
            + 1008 * (x2) * (y6)
            - 252 * (y8)
            + 120 * x10
            - 360 * (x8) * (y2)
            - 1680 * (x6) * (y4)
            - 1680 * (x4) * (y6)
            - 360 * (x2) * (y8)
            + 120 * y10
        ),  # m = 4     n = 10
        64: c[63]
        * (
            -28 * x6
            + 420 * (x4) * (y2)
            - 420 * (x2) * (y4)
            + 28 * y6
            + 72 * x8
            - 1008 * (x6) * (y2)
            + 1008 * (x2) * (y6)
            - 72 * y8
            - 45 * x10
            + 585 * (x8) * (y2)
            + 630 * (x6) * (y4)
            - 630 * (x4) * (y6)
            - 585 * (x2) * (y8)
            + 45 * y10
        ),  # m = 6    n = 10
        65: c[64]
        * (
            -9 * x8
            + 252 * (x6) * (y2)
            - 630 * (x4) * (y4)
            + 252 * (x2) * (y6)
            - 9 * y8
            + 10 * x10
            - 270 * (x8) * (y2)
            + 420 * (x6) * (y4)
            + 420 * (x4) * (y6)
            - 270 * (x2) * (y8)
            + 10 * y10
        ),  # m = 8    n = 10
        66: c[65]
        * (
            -1 * x10
            + 45 * (x8) * (y2)
            - 210 * (x6) * (y4)
            + 210 * (x4) * (y6)
            - 45 * (x2) * (y8)
            + y10
        ),  # m = 10   n = 10
    }
    return Z
