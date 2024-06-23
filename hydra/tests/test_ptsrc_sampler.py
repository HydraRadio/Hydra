import unittest

import numpy as np
from hydra import ptsrc_sampler, example, utils

class TestPtsrcSampler(unittest.TestCase):

    def test_proj_operator_and_precompute_mpi(self):
        # Tests two interdependent functions in one
        import pyuvsim

        Nptsrc = 17

        # Basic array layout
        ant_pos = {
                        0: (0., 0., 0.),
                        1: (0., 14., 0.),
                        2: (0., 0., 14.),
                   }
        antpairs = [(0,1), (0,2), (1,2)]
        lsts = np.linspace(0., 3., 5) # LSTs
        beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
                         for ant in ant_pos.keys()]

        # Settings
        freqs = np.linspace(100., 200., 10) # MHz


        # RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
        ra, dec, amps = example.generate_random_ptsrc_catalogue(
                                        Nptsrc=Nptsrc, 
                                        ra_bounds=(0., 2.*np.pi), 
                                        dec_bounds=(-0.5*np.pi, 0.5*np.pi), 
                                        logflux_bounds=(-1.0, 2.0)
                                        )
        
        # Get fluxes from ptsrc amplitude at ref. frequency
        beta_ptsrc = -2.7
        fluxes = utils.get_flux_from_ptsrc_amp(amps, freqs, beta_ptsrc)

        proj = ptsrc_sampler.calc_proj_operator(
                                        ra,
                                        dec,
                                        fluxes,
                                        ant_pos,
                                        antpairs,
                                        freqs,
                                        lsts,
                                        beams,
                                        latitude=-0.5361913261514378,
                                    )
        self.assertTrue(np.all(~np.isnan(proj)))

        # Pretend data chunk
        x_ptsrc = np.zeros(Nptsrc)
        data_shape = (len(antpairs), freqs.size, lsts.size)
        data = (  proj.reshape((-1, Nptsrc)) @ (1. + x_ptsrc) ).reshape(data_shape)
        gains = (1.1 + 1.0j) * np.ones((len(ant_pos), freqs.size, lsts.size))

        # Check that function runs
        linear_op, linear_rhs = ptsrc_sampler.precompute_mpi(
                                                comm=None,
                                                ants=list(ant_pos.keys()),
                                                antpairs=antpairs,
                                                freq_chunk=freqs,
                                                time_chunk=lsts,
                                                proj_chunk=proj,
                                                data_chunk=data,
                                                inv_noise_var_chunk=np.ones_like(data),
                                                gain_chunk=gains,
                                                amp_prior_std=0.1*np.ones(Nptsrc),
                                                realisation=True
                                                )
        self.assertTrue(np.all(~np.isnan(linear_op)))
        self.assertTrue(np.all(~np.isnan(linear_rhs)))
        



