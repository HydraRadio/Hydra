import numpy as np
import scipy as sp
from scipy.stats import mode
from scipy.signal.windows import blackmanharris as BH
from scipy.stats import invgamma
from scipy.optimize import minimize, Bounds

# from multiprocess import Pool
from . import utils
import os, time


def sample_S(s=None, sk=None, prior=None):
    """
    Draw samples of the bandpowers of S, p(S|s). This assumes that the conditional
    distributions for the bandpowers are uncorrelated with one another, i.e. the Fourier-
    space covariance S has no off-diagonals.

    Parameters:
        s (array_like):
            A set of real-space samples of the field, of shape
            `(Ntimes, Nfreq)`. This will be Fourier transformed.
            Alternatively, `sk` can be given.

        sk (array_like):
            A set of Fourier-space samples of the field, of shape
            `(Ntimes, Nfreq)`.  The monopole is expected to be at the center
            of the frequency axis, i.e. the frequency axis has been fftshifted.

        prior (array_like):
            Array of delta function prior values, used to set certain modes to a
            fixed value.
    """
    if s is None and sk is None:
        raise ValueError("Must pass in s (real space) or sk (Fourier space) vector.")

    if sk is None:
        axes = (1,)
        sk = np.fft.ifftshift(s, axes=axes)
        sk = np.fft.fftn(sk, axes=axes)
        sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape

    beta = np.sum(sk * sk.conj(), axis=0).real
    # The shape parameter (alpha) differs from that used in Eriksen et al. 2008
    # i.e. `alpha = Nobs/2 - 1` because our data vector is complex and has
    # twice as many numbers as a purely real data vector
    alpha = Nobs - 1.0

    x = np.zeros(Nfreqs)
    for i in range(Nfreqs):
        x[i] = invgamma.rvs(a=alpha) * beta[i]  # y = x / beta

    # Set prior
    if prior is not None:
        for i in range(Nfreqs):
            if prior[0, i] == 0:
                continue
            else:
                if x[i] > prior[0, i]:
                    x[i] = prior[0, i]
                if x[i] < prior[1, i]:
                    x[i] = prior[1, i]
    return x


def sprior(signals, bins, factor):

    # prior on cov samples

    # bins - number of bins past zero delay to take, either side. e.g. bins=2 takes delays [-2,-1,0,1,2] from centre
    # factor is maximum factor to multiply / divide the truth by
    Nobs, Nfreq = signals.shape

    sk_ = np.fft.fft(signals, axis=-1)
    ds = np.sum(sk_ * sk_.conj(), axis=0).real
    prior = np.zeros((2, Nfreq))

    prior[0] = ds * factor
    prior[1] = ds / factor

    prior[0, bins + 1 : -bins] = 0
    prior[1, bins + 1 : -bins] = 0

    return prior / (Nobs / 2 - 1)


def gcr_fgmodes_1d(
    vis, w, matrices, fgmodes, f0=None, map_estimate=False, verbose=False
):
    """
    Perform the GCR step on a single time sample.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        w (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fgmodes (array_like):
            Foreground mode array, of shape (Nmodes, Nfreqs). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    """
    Nfreqs, Nmodes = fgmodes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))

    # Extract precomputed matrices needed by the linear system
    Sh = matrices[0][0]
    S = matrices[0][1]
    Ni = matrices[0][2]
    Nih = matrices[0][3]
    A = matrices[1][0]
    Ai = matrices[1][1]

    if map_estimate:
        oma = np.zeros((Nfreqs, 1), dtype=complex)
        omb = np.zeros((Nfreqs, 1), dtype=complex)
    else:
        # Unit complex Gaussian random realisation
        omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs, 1)
        omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs, 1)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5

    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    b[:Nfreqs] = S @ Ni @ (w * d).T + Sh @ oma + S @ Nih @ omb
    b[Nfreqs:] = fgmodes.T.conj() @ (Ni @ (w * d).T + Nih @ omb)

    # Run CG solver, preconditioned by M=Ai
    x0 = None
    if f0 is not None:
        x0 = np.concatenate((np.zeros(Nfreqs, dtype=complex), f0))
    xsoln, info = sp.sparse.linalg.cg(A, b, maxiter=1e5, x0=x0, M=Ai)
    if verbose:
        residual = np.abs(A @ xsoln - b[:, 0]).mean()
    else:
        residual = None

    # Return solution vector
    return xsoln, residual, info


def gcr_fgmodes(
    vis, w, matrices, fgmodes, f0=None, nproc=1, map_estimate=False, verbose=False
):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        w (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fgmodes (array_like):
            Foreground mode array, of shape (Nmodes, Nfreqs). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        fourier_op (array_like):
            Pre-computed Fourier operator.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        samples (array_like):
            Array of signal + foreground realisations for each time sample,
            of shape `(Ntimes, Nfreqs + Nmodes)`.
    """
    samples = np.zeros((vis.shape[0], vis.shape[1] + fgmodes.shape[1]), dtype=complex)
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    else:
        residuals = None
        info = None
    idxs = np.arange(vis.shape[0])

    # Run GCR method on each time sample in parallel
    if verbose:
        st = time.time()
    with Pool(nproc) as pool:
        samples, residuals, info = zip(
            *pool.map(
                lambda idx: gcr_fgmodes_1d(
                    vis=vis[idx],
                    w=w,
                    matrices=matrices,
                    fgmodes=fgmodes,
                    f0=f0,
                    map_estimate=map_estimate,
                    verbose=verbose,
                ),
                idxs,
            )
        )
    samples = np.array(samples).reshape((vis.shape[0], -1))
    residuals = np.array(residuals)
    info = np.array(info)

    # Return sample
    if verbose:
        print(f"{time.time() - st:<12.1f}", end="")
        print(f"{info.mean():<8.1f}", end="")
        print(f"{residuals.mean():<12.2e}", end="")
    return samples


def covariance_from_pspec(ps, fourier_op):
    """
    Transform the sampled power spectrum into a frequency-frequency covariance
    matrix that can be used for the next iteration.
    """
    Nfreqs = ps.size
    Csigfft = np.zeros((Nfreqs, Nfreqs), dtype=complex)
    Csigfft[np.diag_indices(Nfreqs)] = ps
    C = fourier_op.T.conj() @ Csigfft @ fourier_op
    return C


def build_matrices(Nparams, flags, signal_S, Ninv, fgmodes):
    """
    Calculate matrices and build A in Ax=b for the GCR step.

    Parameters:
        Nparams (int):
            Number of model parameters.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape
            `(Nfreqs,)`.
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.

    Returns:
        matrices (list of array_like):
            List containing necessary GCR operators (`matrices[0]`) and the
            linear operator A in the GCR Ax=b solve step.
    """
    Nfreqs = signal_S.shape[0]

    # Construct matrix structure
    matrices = [0, 0]
    matrices[0] = np.zeros((4, Nfreqs, Nfreqs), dtype=complex)
    matrices[1] = np.zeros((2, Nparams, Nparams), dtype=complex)

    # Construct necessary operators for GCR
    matrices[0][0] = sp.linalg.sqrtm(signal_S)  # Sh
    matrices[0][1] = signal_S.copy()  # S
    matrices[0][2] = flags.T * Ninv * flags  # Ni # FIXME
    matrices[0][3] = sp.linalg.sqrtm(matrices[0][2])  # Nih

    # Construct operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = np.eye(Nfreqs) + matrices[0][1] @ matrices[0][2]  # 1 + S @ Ni
    A[:Nfreqs, Nfreqs:] = matrices[0][1] @ matrices[0][2] @ fgmodes
    A[Nfreqs:, :Nfreqs] = fgmodes.T.conj() @ matrices[0][2]
    A[Nfreqs:, Nfreqs:] = fgmodes.T.conj() @ matrices[0][2] @ fgmodes

    matrices[1][0] = A
    matrices[1][1] = np.linalg.pinv(A)  # pseudo-inverse, to be used as a preconditioner

    return matrices


def gibbs_step_fgmodes(
    vis,
    flags,
    signal_S,
    fgmodes,
    Ninv,
    ps_prior=None,
    f0=None,
    nproc=1,
    map_estimate=False,
    verbose=False,
):
    """
    Perform a single Gibbs iteration for a Gibbs sampling scheme using a foreground model
    based on frequency templates for multiple foreground modes.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape
            `(Nfreqs,)`.
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        ps_prior (array_like):
            EoR signal power spectrum prior.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Ntimes, Nfreqs)`.
        S_sample (array_like):
            Sample of the signal covariance, shape `(Nfreqs, Nfreqs)`. This is
            simply a transformation of the power spectrum.
        ps_sample (array_like):
            Sample of the signal power spectrum bandpowers, shape `(Nfreqs,)`.
        fg_amps (array_like):
            Sample of the foreground amplitudes, shape `(Nmodes,)`.
    """
    # Shape of data and operators
    Nfreqs = vis.shape[1]
    Nmodes = fgmodes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)

    # Get matrices necessary for the GCR step
    matrices = build_matrices(Nparams, flags, signal_S, Ninv, fgmodes)

    # (1) Solve GCR equation to get EoR signal and foreground amplitude realisations
    cr = gcr_fgmodes(
        vis=vis,
        w=flags,
        matrices=matrices,
        fgmodes=fgmodes,
        f0=f0,
        nproc=nproc,
        map_estimate=map_estimate,
        verbose=verbose,
    )

    # Extract separate signal and FG parts from the solution
    signal_cr = cr[:, : -fgmodes.shape[1]]
    fg_amps = cr[:, -fgmodes.shape[1] :]

    # Full model of data is sum of EoR (GCR) + FG model
    model = (
        signal_cr + fg_amps @ fgmodes.T
    )  # np.einsum('ijk,lk->ijl', fg_amps, fgmodes)
    # Chi-squared is computed as the sum of ( |data - model| / noise )^2,
    # i.e. as a sum of standard normal random variables.
    # FIXME: this will need to be changed to account for time-dependent
    # flags (i.e. when we have a different N per time).
    chisq = np.abs(vis - model) ** 2 * Ninv.diagonal()[None, :]
    if verbose:
        chisq_mean = chisq[:, flags].mean()
        if chisq_mean > 10:
            print(f"{chisq_mean:<9.1e}", end="")
        else:
            print(f"{chisq_mean:<9.3f}", end="")

    # (2) Sample EoR signal power spectrum (and also convert to equivalent
    # covariance matrix sample)
    ps_sample = sample_S(s=signal_cr, prior=ps_prior)
    # The factor of 1/Nfreqs**2 here is an FFT normalization
    S_sample = covariance_from_pspec(ps_sample / Nfreqs**2, fourier_op)

    # Log posterior
    # Each time is treated as an independent sample.  So, the joint
    # log posterior for all times is the sum of the individual log
    # posteriors for each time.
    # WARNING: np.linalg.inv should be avoided for general, dense matrices.
    # S_sample should be diagonally dominant and thus this should be okay.
    Sinv = np.linalg.inv(S_sample)
    ln_post = np.sum(
        np.diagonal(
            -(
                (vis - model)[:, flags].conj()
                @ Ninv[flags][:, flags]
                @ (vis - model)[:, flags].T
            )
            - (
                signal_cr[:, flags].conj()
                @ Sinv[flags][:, flags]
                @ signal_cr[:, flags].T
            )
        )
    )
    ln_post = ln_post.real
    if verbose:
        print(f"{ln_post:<12.1f}")

    # Return samples
    return signal_cr, S_sample, ps_sample, fg_amps, chisq, ln_post
