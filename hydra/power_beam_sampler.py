import numpy as np

from hydra.per_ant_beam_sampler import gen_std_norm

################################################################################
# The construct_LHS and construct_RHS options do a common computation, NinvB. ##
# It could be worth making this a common argument.                            ##
################################################################################
def construct_LHS(pow_beam_Dmatr, Ninv, Cinv):
    """
    Make left hand side operator for GCR based on the design matrix, inverse noise
    covariance, and inverse prior covariance.

    Parameters:
        pow_beam_Dmatr (array):
            Design matrix, raveled into shape (NFREQS, NTIMES, NBASELINES, NBASIS)
        Ninv (array):
            Inverse noise covariance, assumed diagonal, so of shape (NFREQS, NTIMES, NBASELINES)
        Cinv (array):
            Inverse prior covariance at each frequency, of shape (NFREQS, NBASIS, NBASIS.
            Setting to 0 makes the return value equal to the Fisher matrix.
    """
    Nfreqs = Ninv.shape[0]
    Nbasis = pow_beam_Dmatr.shape[-1]
    fisher = np.zeros([Nfreqs, Nbasis, Nbasis], dtype=complex)
    # Ultimately want ftub,ftu,ftuB->fbB
    # Start with ftu,ftuB->ftuB
    NinvB = Ninv[:, :, :, None] * pow_beam_Dmatr
    # Now do ftub,ftuB->fbB
    for freq_ind in range(Nfreqs):
        # tub,tuB->bB
        fisher[freq_ind] = np.tensordot(
            pow_beam_Dmatr[freq_ind].conj(),
            NinvB[freq_ind],
            axes=((0, 1), (0, 1))
        )
    
    LHS = fisher + Cinv

    return LHS

def construct_RHS(pow_beam_Dmatr, Ninv, Cinv, vis, prior_mean, flx=True,
                  rng=None, Cinv_cho=None):
    """
    Construct the RHS of the GCR equation.

    Parameters:
        pow_beam_Dmatr (array):
            Design matrix, raveled into shape (NFREQS, NTIMES, NBASELINES, NBASIS)
        Ninv (array):
            Inverse noise covariance, assumed diagonal, so of shape (NFREQS, NTIMES, NBASELINES)
        Cinv (array):
            Inverse prior covariance at each frequency, of shape (NFREQS, NBASIS, NBASIS.
            Setting to 0 makes the return equal to the Fisher matrix.
        vis (array):
            Complex visibility data, shape (NFREQS, NTIMES, NBASELINES).
        prior_mean (array):
            Mean of the Gaussian prior, shape (NFREQS, NBASIS).
        flx (bool):
            Whether to include the fluctuation term (sampling vs optimizing).
        Cinv_cho (array):
            Cholesky decomposition of the inverse prior covariance such that
            M M^\dag = Cinv, shape (NFREQS, NBASIS, NBASIS).
    """
    Ninvd = Ninv * vis
    Nfreqs = vis.shape[0]
    Nbasis = pow_beam_Dmatr.shape[-1]

    Bdag_Ninvd = np.zeros([Nfreqs, Nbasis], dtype=complex)
    Cinv_mu = np.zeros_like(Bdag_Ninvd)
    if flx:
        if rng is None:
            raise ValueError("Must pass valid rng instance if sampling.")
        flx_vis = np.zeros_like(Bdag_Ninvd)
        flx_prior = np.zeros_like(Bdag_Ninvd)

        # TODO: Check that rng's state gets updated inside of gen_std_norm
        # Otherwise parts of these draws will be identical
        rnd_vis = np.sqrt(Ninv) * gen_std_norm(vis.shape, rng)
        rnd_prior = gen_std_norm(flx_prior.shape, rng)
    # ftub,ftu->fb
    for freq_ind in range(Nfreqs):
        # tub,tu->b
        Bdag_Ninvd[freq_ind] = np.tensordot(
            Ninvd[freq_ind],
            pow_beam_Dmatr[freq_ind].conj(),
            axes=2
        )
        Cinv_mu[freq_ind] = np.tensordot(
            Cinv[freq_ind], 
            prior_mean[freq_ind], 
            axes=1
        )

        if flx:
            # tub,tu->b
            flx_vis[freq_ind] = np.tensordot(
                rnd_vis[freq_ind],
                pow_beam_Dmatr[freq_ind].conj(),
                axes=2
            )

            flx_prior[freq_ind] = np.tensordot(
                Cinv_cho[freq_ind],
                rnd_prior[freq_ind],
                axes=1
            )
    RHS = Bdag_Ninvd + Cinv_mu
    if flx:
        RHS += flx_vis + flx_prior

    return RHS
    