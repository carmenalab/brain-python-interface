#!/usr/bin/python
'''
Equivalence test for changes to vfb training procedure (riglib.bmi.train._train_KFDecoder_visual_feedback)
'''
import numpy as np
from db import dbfunctions as dbfn
from db.tracker import models
from riglib.bmi import kfdecoder, ppfdecoder
from riglib.bmi import train
from riglib.bmi import extractor
import unittest

reload(train)
reload(kfdecoder)
reload(ppfdecoder)
reload(extractor)

class TestDecoderTrain(unittest.TestCase):
    def test_kalman_vf(self):
        te = dbfn.TaskEntry(2424, dbname='testing')
        dec_record = te.decoder_record
        dec = dec_record.load()
      
        training_block = dbfn.TaskEntry(dec_record.entry_id, dbname='testing')
        files = training_block.datafiles
        print files
      
        from riglib.bmi import extractor
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict()
        extractor_kwargs['n_subbins'] = dec.n_subbins 

        dec_new = train.train_KFDecoder(files, extractor_cls, extractor_kwargs, train.get_plant_pos_vel, train.endpt_2D_state_space, dec.units, update_rate=dec.binlen, kin_source='task', pos_key='cursor', vel_key=None, tslice=dec.tslice)
        for attr in dec.filt.model_attrs:
            old_attr = getattr(dec.filt, attr)
            new_attr = getattr(dec_new.filt, attr)
            self.assertTrue(np.all((old_attr - new_attr) < 1e-10))

        self.assertTrue(np.array_equal(dec_new.units, dec.units))
        self.assertTrue(np.array_equal(dec_new.bounding_box[0], dec.bounding_box[0]))
        self.assertTrue(np.array_equal(dec_new.bounding_box[1], dec.bounding_box[1]))
        self.assertTrue(dec_new.states == dec.states)
        self.assertTrue(dec_new.states_to_bound == dec.states_to_bound)
        self.assertTrue(np.array_equal(dec_new.mFR, dec.mFR))
        self.assertTrue(np.array_equal(dec_new.sdFR, dec.sdFR))
        self.assertTrue(np.array_equal(dec_new.drives_neurons, dec.drives_neurons))

        #self.assertTrue(dec.filt == dec_new.filt)

    def test_ppf_vf(self):
        te = dbfn.TaskEntry(2425, dbname='testing')
        dec_record = te.decoder_record
        dec = dec_record.load()
        
        training_block = dbfn.TaskEntry(dec_record.entry_id, dbname='testing')
        files = training_block.datafiles
        
        from riglib.bmi import extractor
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict()
        extractor_kwargs['units'] = dec.units
        extractor_kwargs['n_subbins'] = dec.n_subbins 

        dec_new = train.train_PPFDecoder(files, extractor_cls, extractor_kwargs, train.get_plant_pos_vel, train.endpt_2D_state_space, dec.units, update_rate=dec.binlen, kin_source='task', pos_key='cursor', vel_key=None, tslice=dec.tslice)        
        for attr in dec.filt.model_attrs:
            old_attr = getattr(dec.filt, attr)
            new_attr = getattr(dec_new.filt, attr)
            self.assertTrue(np.all((old_attr - new_attr) < 1e-10))

        self.assertTrue(np.array_equal(dec_new.units, dec.units))
        self.assertTrue(np.array_equal(dec_new.bounding_box[0], dec.bounding_box[0]))
        self.assertTrue(np.array_equal(dec_new.bounding_box[1], dec.bounding_box[1]))
        self.assertTrue(dec_new.states == dec.states)
        self.assertTrue(dec_new.states_to_bound == dec.states_to_bound)
        self.assertTrue(np.array_equal(dec_new.drives_neurons, dec.drives_neurons))

        # self.assertTrue(dec.filt == dec_new.filt)


if __name__ == '__main__':
    unittest.main()
