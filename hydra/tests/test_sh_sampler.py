
import unittest

import numpy as np
from hydra import sh_sampler

class TestRegionSampler(unittest.TestCase):

    def test_get_em_ell_idx(self):

        # Check that the function runs
        lmax = 10
        ems, ells, idxs = sh_sampler.get_em_ell_idx(lmax)

        # Check that the right number of m modes exist
        # There should be lmax + 1 modes with m=0. All of the real ones are 
        # retained, but none of the imaginary ones.
        ems = np.array(ems)
        self.assertEqual(ems[ems == 0].size, lmax + 1)

    def test_vis_proj_operator_no_rot(self):

        # Set up for test
        import pyuvsim

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
        lmax = 10
        nside = 8
        freqs = np.linspace(100., 200., 10) # MHz

        # Check that the function runs
        ems, ells, idxs = sh_sampler.get_em_ell_idx(lmax)
        proj, autos, ell, m = sh_sampler.vis_proj_operator_no_rot(
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
                                    )

        # Check the shape of the returned projection operator 
        Nbls = len(antpairs)
        self.assertEqual(proj.shape, (Nbls * freqs.size * lsts.size, len(ems)))

        # Check that operator is finite
        self.assertTrue(np.all(~np.isnan(proj)))

        # Check that the operator is complex, i.e. doesn't have lots of 
        # imaginary parts close to zero
        self.assertTrue(~np.allclose(proj.imag, np.zeros(proj.shape)))

        

    def test_alms_healpy_gsm(self):
        # Test several related functions in one block
        
        # Settings
        lmax = 10
        nside = 8
        freqs = np.linspace(100., 200., 10) # MHz

        # Get alms in hydra format (without modes that are zero)
        alms = sh_sampler.get_alms_from_gsm(freqs, 
                                            lmax, 
                                            nside=nside, 
                                            resolution="low", 
                                            output_model=False, 
                                            output_map=False)

        # Get alms in healpy format
        healpy_alms = sh_sampler.get_healpy_from_gsm(freqs, 
                                                     lmax, 
                                                     nside=nside, 
                                                     resolution="low", 
                                                     output_model=False, 
                                                     output_map=False)
        
        # Convert healpy alms to hydra format and vice versa
        converted_alms = sh_sampler.healpy2alms(healpy_alms)
        converted_healpy_alms = sh_sampler.alms2healpy(alms, lmax)
        
        # Check that results are finite
        self.assertTrue(np.all(~np.isnan(alms)))
        self.assertTrue(np.all(~np.isnan(healpy_alms)))
        self.assertTrue(np.all(~np.isnan(converted_alms)))
        self.assertTrue(np.all(~np.isnan(converted_healpy_alms)))
        self.assertTrue(np.all(~np.isinf(alms)))
        self.assertTrue(np.all(~np.isinf(healpy_alms)))
        self.assertTrue(np.all(~np.isinf(converted_alms)))
        self.assertTrue(np.all(~np.isinf(converted_healpy_alms)))

        # Check that outputs match
        self.assertTrue(np.allclose(alms, converted_alms))
        self.assertTrue(np.allclose(healpy_alms, converted_healpy_alms))

        # Check data types
        self.assertEqual(alms.dtype, np.float64)
        self.assertEqual(converted_alms.dtype, np.float64)
        self.assertEqual(healpy_alms.dtype, np.complex128)
        self.assertEqual(converted_healpy_alms.dtype, np.complex128)
        

    def test_construct_rhs_no_rot(self):
        pass
        #construct_rhs_no_rot(data, inv_noise_var, inv_prior_var, omega_0, omega_1, a_0, vis_response)
        #construct_rhs_no_rot_mpi(comm, data, inv_noise_var, inv_prior_var, omega_a, omega_n, a_0, vis_response)

    def test_apply_lhs(self):
        pass
        #apply_lhs_no_rot_mpi(comm, a_cr, inv_noise_var, inv_prior_var, vis_response)
        #apply_lhs_no_rot(a_cr, inv_noise_var, inv_prior_var, vis_response)
    
    def test_radiometer_eq(self):
        """
        radiometer_eq(auto_visibilities, 
                      ants, delta_time, 
                      delta_freq, 
                      Nnights=1, 
                      include_autos=False)
        """
        pass