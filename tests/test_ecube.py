import time
from riglib import source
from riglib.ecube import Broadband, LFP, LFP_Blanking_File, LFP_Blanking, make_source_class
from riglib.ecube.file import parse_file
from riglib.bmi import state_space_models, train, extractor
import numpy as np
import unittest

import unittest

STREAMING_DURATION = 3

class TestStreaming(unittest.TestCase):

    # Note: for the multiprocessing code to work, you must have servernode-control open separately

    @unittest.skip('works')
    def test_ecube_stream(self):
        channels = [1, 3]
        bb = Broadband(channels=channels) # Note: this is not how this is normally used, just testing
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

    @unittest.skip('works')
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
    
    @unittest.skip('works')
    def test_ds_with_extractor(self):
        # Create the datasource
        channels = [1, 62]
        ds = source.MultiChanDataSource(LFP, channels=channels)

        # Make a feature extractor
        extr = extractor.LFPMTMPowerExtractor(ds, channels=channels, bands=[(90,110)], win_len=0.1, fs=1000)

        # Run the feature extractor
        extract_rate = 1/60
        feats = []
        ds.start()
        t0 = time.perf_counter()
        while (time.perf_counter() - STREAMING_DURATION < t0):
            neural_features_dict = extr(time.perf_counter())
            feats.append(neural_features_dict['lfp_power'])
            time.sleep(extract_rate)
        
        ds.stop()
        data = np.array(feats)
        print(feats)
        print(data.shape)
        self.assertEqual(data.shape[1], len(channels))
        self.assertEqual(data.shape[0], STREAMING_DURATION/extract_rate)

    def test_lfp_blanking(self):
        channels = [1, 3]
        file = "/media/server/raw/ecube/2022-12-23_BMI3D_te7797"
        bb = LFP_Blanking_File(channels=channels, ecube_bmi_filename=file, trig_channel=0) # Note: this is not how this is normally used, just testing
        bb.start()
        ch_data = []
        chs = []
        for d in range(100):
            for i in range(len(channels)):
                ch, data = bb.get()
                ch_data.append(data)
                chs.append(ch)
                if np.count_nonzero(np.isnan(data)):
                    print(f"Got some nans in ch {ch}")
        print(np.unique(chs))
        bb.stop()


class TestFileLoadin(unittest.TestCase):

    def test_ecube_load(self):
        pass


    def extract_from_file_lfp(self):

        pass

        #LFPMTMPowerExtractor.extract_from_file()

if __name__ == '__main__':
    unittest.main()