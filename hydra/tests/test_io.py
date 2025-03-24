
import unittest

import numpy as np
from hydra import io

class TestIO(unittest.TestCase):

    def test_load_uvdata_metadata(self):

        import pyuvdata
        import os

        # Get path to pyuvdata's built-in test data
        fname = os.path.join(pyuvdata.data.DATA_PATH, "zen.2458432.34569.uvh5")

        # Check that the function runs
        data_info = io.load_uvdata_metadata(comm=None, fname=fname)

        # Check that returned dict has the expected fields
        expected_fields = ['freqs', 'lsts', 'lat', 'lon', 'alt', 'antpairs', 
                           'ants1', 'ants2', 'ants']
        for f in expected_fields:
            self.assertTrue(f in data_info.keys())
        self.assertTrue(len(data_info) == len(expected_fields))

        # Check that the data can be loaded
        data, flags = io.partial_load_uvdata(fname, 
                                             freq_chunk=data_info['freqs'], 
                                             lst_chunk=data_info['lsts'], 
                                             antpairs=data_info['antpairs'])
        