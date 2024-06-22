
import unittest

import numpy as np
from hydra import region_sampler

class TestRegionSampler(unittest.TestCase):

    def test_get_diffuse_sky_model_pixels(self):

        freqs = np.linspace(100., 200., 10) # MHz

        # Check that each model type runs
        ra1, dec1, sky_maps1 = region_sampler.get_diffuse_sky_model_pixels(freqs, 
                                                                        nside=8, 
                                                                        sky_model='gsm2008')
        ra2, dec2, sky_maps2 = region_sampler.get_diffuse_sky_model_pixels(freqs, 
                                                                        nside=8, 
                                                                        sky_model='gsm2016')
        ra3, dec3, sky_maps3 = region_sampler.get_diffuse_sky_model_pixels(freqs, 
                                                                        nside=8, 
                                                                        sky_model='lfsm')
        ra4, dec4, sky_maps4 = region_sampler.get_diffuse_sky_model_pixels(freqs, 
                                                                        nside=8, 
                                                                        sky_model='haslam')

        # Make sure returned dimensions are correct
        self.assertEqual(ra1.size, ra2.size)
        self.assertEqual(ra1.size, ra3.size)
        self.assertEqual(ra1.size, ra4.size)
        self.assertEqual(ra1.size, dec1.size)
        self.assertEqual(dec1.size, dec2.size)
        self.assertEqual(dec1.size, dec3.size)
        self.assertEqual(dec1.size, dec4.size)

        # Make sure sky_maps have dimensions (Nfreqs, Npix)
        self.assertEqual(ra1.size, sky_maps1.shape[0])
        self.assertEqual(ra2.size, sky_maps2.shape[0])
        self.assertEqual(ra3.size, sky_maps3.shape[0])
        self.assertEqual(ra4.size, sky_maps4.shape[0])
        self.assertEqual(freqs.size, sky_maps1.shape[1])
        self.assertEqual(freqs.size, sky_maps2.shape[1])
        self.assertEqual(freqs.size, sky_maps3.shape[1])
        self.assertEqual(freqs.size, sky_maps4.shape[1])
        healpix_npix = 12 * 8**2 # (Npix = 12 * Nside^2)
        self.assertEqual(ra1.size, healpix_npix)

        # Check that no pixels are NaN
        self.assertTrue(np.all(~np.isnan(ra1)))
        self.assertTrue(np.all(~np.isnan(sky_maps1)))
        self.assertTrue(np.all(~np.isnan(sky_maps2)))
        self.assertTrue(np.all(~np.isnan(sky_maps3)))
        self.assertTrue(np.all(~np.isnan(sky_maps4)))

        # Check RA and Dec bounds
        print(np.rad2deg(dec1).astype(int))
        self.assertTrue( np.all(np.logical_and(ra1 >= np.deg2rad(0.), ra1 < np.deg2rad(360.))) )
        self.assertTrue( np.all(np.logical_and(ra2 >= np.deg2rad(0.), ra2 < np.deg2rad(360.))) )
        self.assertTrue( np.all(np.logical_and(ra3 >= np.deg2rad(0.), ra3 < np.deg2rad(360.))) )
        self.assertTrue( np.all(np.logical_and(ra4 >= np.deg2rad(0.), ra4 < np.deg2rad(360.))) )
        self.assertTrue( np.all(np.logical_and(dec1 >= np.deg2rad(-90.), dec1 <= np.deg2rad(90.))) )
        self.assertTrue( np.all(np.logical_and(dec2 >= np.deg2rad(-90.), dec2 <= np.deg2rad(90.))) )
        self.assertTrue( np.all(np.logical_and(dec3 >= np.deg2rad(-90.), dec3 <= np.deg2rad(90.))) )
        self.assertTrue( np.all(np.logical_and(dec4 >= np.deg2rad(-90.), dec4 <= np.deg2rad(90.))) )
        
        # Check that higher-res works too
        ra2a, dec2a, sky_maps2a = region_sampler.get_diffuse_sky_model_pixels(freqs, 
                                                                              nside=32, 
                                                                              sky_model='gsm2016')
        healpix_npix = 12 * 32**2 # (Npix = 12 * Nside^2)
        self.assertEqual(ra2a.size, healpix_npix)




if __name__ == "__main__":
    unittest.main()