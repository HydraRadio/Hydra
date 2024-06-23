
import unittest

import numpy as np
from hydra import example, utils

class TestExample(unittest.TestCase):

    def test_example_functions(self):
        # Tests two interdependent functions in one
        import pyuvsim

        Nptsrc = 17

        # Basic array layout
        lsts = np.linspace(0., 3., 5) # LSTs

        # Settings
        freqs = np.linspace(100., 200., 10) # MHz

        # Check that function runs
        # RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
        ra, dec, amps = example.generate_random_ptsrc_catalogue(
                                        Nptsrc=Nptsrc, 
                                        ra_bounds=(0., 2.*np.pi), 
                                        dec_bounds=(-0.5*np.pi, 0.5*np.pi), 
                                        logflux_bounds=(-1.0, 2.0)
                                        )
        
        # Check RA, Dec, and amplitude bounds match what was input
        self.assertTrue( np.all(np.logical_and(ra >= np.deg2rad(0.), ra < np.deg2rad(360.))) )
        self.assertTrue( np.all(np.logical_and(dec >= np.deg2rad(-90.), dec <= np.deg2rad(90.))) )
        self.assertTrue( np.all(np.logical_and(amps >= 1e-1, amps <= 1e2)) )
        self.assertTrue(np.all(~np.isnan(ra)))
        self.assertTrue(np.all(~np.isnan(dec)))
        self.assertTrue(np.all(~np.isnan(amps)))

        # Check that function runs
        model0_chunk, fluxes, beams, ant_info = example.run_example_simulation( 
                                                           output_dir="/tmp/_hydra_example_sim/", 
                                                           times=lsts,
                                                           freqs=freqs,
                                                           ra=ra, 
                                                           dec=dec, 
                                                           ptsrc_amps=amps,
                                                           hex_array=(3,4),
                                                           beam_type='gaussian', 
                                                           array_latitude=np.deg2rad(-30.7))
        ants, ant_pos, antpairs, ants1, ants2 = ant_info

        self.assertTrue(np.all(~np.isnan(model0_chunk)))
        self.assertTrue(np.all(~np.isnan(fluxes)))
        self.assertEqual(model0_chunk.shape, (len(antpairs), freqs.size, lsts.size))
        self.assertEqual(fluxes.shape, (Nptsrc, freqs.size))
        self.assertEqual(len(beams), len(ant_pos))

        Nants = 10 # (3, 4, 3) hex array
        self.assertEqual(len(antpairs), Nants * (Nants - 1) // 2)

