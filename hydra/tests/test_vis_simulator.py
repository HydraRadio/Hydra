
import unittest

import numpy as np
from hydra import vis_simulator, utils, example

class TestVisSimulator(unittest.TestCase):

    def test_vis_sim_per_source(self):

        import pyuvsim
        from matvis import conversions

        Nptsrc = 17

        # Basic array layout
        ant_pos = {
                        0: (0., 0., 0.),
                        1: (0., 14., 0.),
                        2: (0., 0., 14.),
                   }
        antpairs = [(0,1), (0,2), (1,2)]
        lsts = np.linspace(0., 3., 5) # LSTs

        # Make dish diameter small so not all sources are far from mainlobe
        beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=6.)
                         for ant in ant_pos.keys()]
        beams = [
            conversions.prepare_beam(beam, polarized=False, use_feed='x')
            for beam in beams
        ]

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

        # Source coordinate transform, from equatorial to Cartesian
        crd_eq = conversions.point_source_crd_eq(ra, dec)

        # Get coordinate transforms as a function of LST
        latitude = np.deg2rad(-30.7215)
        eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])
        antpos = np.array([ant_pos[k] for k in ant_pos.keys()])

        for j, freq in enumerate(freqs):

            # Run vis_sim_per_source in mode that includes sqrt(fluxes) in calculation
            vis1 = vis_simulator.vis_sim_per_source(
                                    antpos,
                                    freq*1e6,
                                    eq2tops,
                                    crd_eq,
                                    fluxes[:,j],
                                    beam_list=beams,
                                    precision=2,
                                    polarized=False,
                                    force_no_beam_sqrt=True,
                                    apply_fluxes_afterwards=False,
                                )
            self.assertTrue(np.all(~np.isnan(vis1)))

            # Run vis_sim_per_source in mode that applies fluxes after calculating 
            # the visibility response
            vis1a = vis_simulator.vis_sim_per_source(
                                    antpos,
                                    freq*1e6,
                                    eq2tops,
                                    crd_eq,
                                    fluxes[:,j],
                                    beam_list=beams,
                                    precision=2,
                                    polarized=False,
                                    force_no_beam_sqrt=True,
                                    apply_fluxes_afterwards=True,
                                )
            self.assertTrue(np.all(~np.isnan(vis1a)))

            # Both methods should match
            self.assertTrue(np.allclose(vis1, vis1a))