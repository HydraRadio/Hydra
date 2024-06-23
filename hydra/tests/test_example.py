
import unittest

import numpy as np
from hydra import example

"""
    ra, dec, ptsrc_amps = example.generate_random_ptsrc_catalogue(Nptsrc, 
                                                              ra_bounds=args.ra_bounds, 
                                                              dec_bounds=args.dec_bounds, 
                                                              logflux_bounds=(-1., 2.))
    
    model0_chunk, fluxes_chunk, beams, ant_info = example.run_example_simulation(
                                                       args=args, 
                                                       output_dir=output_dir, 
                                                       times=time_chunk,
                                                       freqs=freq_chunk,
                                                       ra=ra, 
                                                       dec=dec, 
                                                       ptsrc_amps=ptsrc_amps,
                                                       array_latitude=array_latitude)
"""