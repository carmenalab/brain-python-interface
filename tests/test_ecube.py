import time
from riglib import source
from riglib.ecube import Broadband, LFP
from riglib.ecube.file import parse_file
import numpy as np

import unittest

STREAMING_DURATION = 1

class TestStreaming(unittest.TestCase):

    @unittest.skip('mst')
    def test_ecube_stream(self):
        channels = [1, 3]
        bb = Broadband(channels=channels)
        bb.start()
        data = bb.conn.get()
        bb.stop()
        print(data[2].shape)
    
    def test_broadband_class(self):
        channels = [1, 3]
        bb = Broadband(channels=channels)
        bb.start()
        ch_data = []
        for i in range(len(channels)):
            ch, data = bb.get()
            ch_data.append(data)
            print(f"Got channel {channels[i]} with {ch_data[-1].shape} samples")
        bb.stop()

    @unittest.skip('mst')
    def test_broadband_datasource(self):
        channels = [1, 62]
        ds = source.MultiChanDataSource(Broadband, channels=channels)
        ds.start()
        time.sleep(STREAMING_DURATION)
        n_samples = int(Broadband.update_freq*STREAMING_DURATION)
        data = ds.get(n_samples, channels)
        ds.stop()

        self.assertEqual(data.shape[1], n_samples)
        self.assertEqual(data.shape[0], len(channels))

class TestFileLoadin(unittest.TestCase):

    def test_ecube_load(self):
        pass


    def extract_from_file_lfp(self):

        pass

        #LFPMTMPowerExtractor.extract_from_file()

if __name__ == '__main__':
    unittest.main()