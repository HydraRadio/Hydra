
import unittest

import numpy as np
from hydra import gain_sampler

class TestGainSampler(unittest.TestCase):

    def test_proj_operator(self):
        
        np.random.seed(1)

        # Basic array layout
        ant_pos = {
                        0: (0., 0., 0.),
                        1: (0., 14., 0.),
                        2: (0., 0., 14.),
                   }
        antpairs = [(0,1), (0,2), (1,2)]
        ants = list(ant_pos.keys())

        # Dimensions of data
        Nbls = len(antpairs)
        Nfreqs = 10
        Ntimes = 20

        # Check that function runs
        A_real, A_imag = gain_sampler.proj_operator(ants, antpairs)
        
        

        model_vis = np.ones((Nbls, Nfreqs, Ntimes), dtype=np.complex128)
        model_vis += np.random.randn(*model_vis.shape) \
                   + 1.j*np.random.randn(*model_vis.shape)
        
        x = np.ones((len(ants), Nfreqs, Ntimes))
        y = gain_sampler.apply_proj(x, A_real, A_imag, model_vis)
        yconj = gain_sampler.apply_proj_conj(model_vis, A_real, A_imag, model_vis, gain_shape=x.shape)
    
    """
    construct_rhs_mpi(
                comm,
                resid,
                inv_noise_var,
                pspec_sqrt,
                A_real,
                A_imag,
                model_vis,
                Fbasis,
                realisation=True,
                seed=None)

    apply_operator_mpi(comm, x, inv_noise_var, pspec_sqrt, A_real, A_imag, model_vis, Fbasis)
    """