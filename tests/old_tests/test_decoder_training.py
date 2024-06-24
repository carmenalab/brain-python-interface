import unittest
from riglib.bmi import train
from db import dbfunctions
import numpy as np

class TestDecoderTraining(unittest.TestCase):
    def test_unit_conv_mm_to_cm(self):

        # block with fixed decoder operating in mm
        te = dbfunctions.get_task_entry(1762)
        
        hdf = dbfunctions.get_hdf(te)
        tslice = slice(5, None, 6)
        cursor = hdf.root.task[tslice]['cursor']
        spike_counts = hdf.root.task[tslice]['bins']
        spike_counts = np.array(spike_counts, dtype=np.float64)
        #spike_counts = spike_counts[5::6] # weird indexing feature of the way the old BMI was running
        
        def run_decoder(dec, spike_counts):
            T = spike_counts.shape[0]
            decoded_state = []
            for t in range(0, T):
                decoded_state.append(dec.predict(spike_counts[t,:]))
            return np.array(np.vstack(decoded_state))
        
        dec = dbfunctions.get_decoder(te)
        dec_state_mm = 0.1*run_decoder(dec, spike_counts)
        diff_mm = cursor - np.float32(dec_state_mm[:,0:3])
        self.assertEqual(np.max(np.abs(diff_mm)), 0)
        
        dec = dbfunctions.get_decoder(te)
        dec_cm = train.rescale_KFDecoder_units(dec)
        dec_state_cm = run_decoder(dec_cm, spike_counts)
        diff_cm = cursor - np.float32(dec_state_cm[:,0:3])
        #print np.max(np.abs(diff_cm))
        self.assertEqual(np.max(np.abs(diff_cm)), 0)

    def test_training_from_mc(self):
        block = 'cart20130815_01'
        files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
        binlen = 0.1
        tslice = [1., 300.]
        
        decoder = train._train_KFDecoder_manual_control(
            cells=None, binlen=binlen, tslice=tslice,
            state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
            **files) 
    
    def test_training_from_bc(self):
        block = 'cart20130521_04'
        files = dict(hdf='/storage/rawdata/hdf/%s.hdf' % block)
        binlen = 0.1
        tslice = [1., 300.]
    
        decoder = train._train_KFDecoder_manual_control(
            cells=None, binlen=binlen, tslice=tslice,
            state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
            **files) 

if __name__ == '__main__':
    unittest.main()
