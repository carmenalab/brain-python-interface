import time
from riglib import source
from riglib.ecube import Broadband, LFP
from riglib.ecube.file import parse_file
import numpy as np

import unittest

STREAMING_DURATION = 2

class TestStreaming(unittest.TestCase):

    # Note: for the multiprocessing code to work, you must have servernode-control open separately

    @unittest.skip('works')
    def test_ecube_stream(self):
        channels = [1, 3]
        bb = Broadband(channels=channels)
        bb.start()
        data = bb.conn.get()
        bb.stop()
        print(data[2].shape)
    
    @unittest.skip('works')
    def test_broadband_class(self):
        channels = [1, 3]
        bb = Broadband(channels=channels)
        bb.start()
        ch_data = []
        for d in range(2):
            for i in range(len(channels)):
                ch, data = bb.get()
                ch_data.append(data)
                print(f"Got channel {ch} with {data.shape} samples")
        bb.stop()

    #@unittest.skip('works')
    def test_broadband_datasource(self):
        channels = [1, 62]
        ds = source.MultiChanDataSource(Broadband, channels=channels)
        ds.start()
        time.sleep(STREAMING_DURATION)
        data = ds.get_new(channels)
        ds.stop()
        data = np.array(data)

        n_samples = int(Broadband.update_freq * STREAMING_DURATION / 728) * 728 # closest multiple of 728 (floor)

        self.assertEqual(data.shape[0], len(channels))
        self.assertEqual(data.shape[1], n_samples)

class TestFileLoadin(unittest.TestCase):

    def test_ecube_load(self):
        pass


    def extract_from_file_lfp(self):

        pass

        #LFPMTMPowerExtractor.extract_from_file()

if __name__ == '__main__':
    unittest.main()