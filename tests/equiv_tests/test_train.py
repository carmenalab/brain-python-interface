#!/usr/bin/python
'''
Equivalence test for changes to vfb training procedure (riglib.bmi.train._train_KFDecoder_visual_feedback)
'''
import numpy as np
from db import dbfunctions as dbfn
from db.tracker import models
from riglib.bmi import train, kfdecoder, ppfdecoder
import unittest

reload(train)
reload(kfdecoder)
reload(ppfdecoder)

class TestDecoderTrain(unittest.TestCase):
    def test_kalman_vf(self):
        te = dbfn.get_task_entry(2424)
        dec_record = dbfn.get_decoder_entry(te)
        dec = dbfn.get_decoder(te)
        
        training_block = dbfn.get_task_entry(dec_record.entry_id)
        datafiles = models.DataFile.objects.filter(entry_id=training_block.id)
        files = dict((d.system.name, d.get_path()) for d in datafiles)
        
        from riglib.bmi import extractor
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict()
        extractor_kwargs['units'] = dec.units
        extractor_kwargs['n_subbins'] = dec.n_subbins 

        # dec_new = train._train_KFDecoder_visual_feedback(cells=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
        dec_new = train._train_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
        # self.assertTrue(np.all((dec.filt.C - dec_new.filt.C) < 1e-10))
        self.assertTrue(np.all((dec.filt.C - dec_new.kf.C) < 1e-10))

        self.assertTrue(np.array_equal(dec_new.units, dec.units))
        self.assertTrue(np.array_equal(dec_new.bounding_box[0], dec.bounding_box[0]))
        self.assertTrue(np.array_equal(dec_new.bounding_box[1], dec.bounding_box[1]))
        self.assertTrue(dec_new.states == dec.states)
        self.assertTrue(dec_new.states_to_bound == dec.states_to_bound)
        self.assertTrue(np.array_equal(dec_new.mFR, dec.mFR))
        self.assertTrue(np.array_equal(dec_new.sdFR, dec.sdFR))
        self.assertTrue(np.array_equal(dec_new.drives_neurons, dec.drives_neurons))

        #self.assertTrue(dec.filt == dec_new.filt)




    # def test_ppf_vf(self):
    #     te = dbfn.get_task_entry(2425)
    #     dec_record = dbfn.get_decoder_entry(te)
    #     dec = dbfn.get_decoder(te)
        
    #     training_block = dbfn.get_task_entry(dec_record.entry_id)
    #     datafiles = models.DataFile.objects.filter(entry_id=training_block.id)
    #     files = dict((d.system.name, d.get_path()) for d in datafiles)
        
    #     from riglib.bmi import extractor
    #     extractor_cls = extractor.BinnedSpikeCountsExtractor
    #     extractor_kwargs = dict()
    #     extractor_kwargs['units'] = dec.units
    #     extractor_kwargs['n_subbins'] = dec.n_subbins 

    #     # dec_new = train._train_PPFDecoder_visual_feedback(cells=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
    #     dec_new = train._train_PPFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
    #     self.assertTrue(np.all((dec.filt.C - dec_new.filt.C) < 1e-10))

    #     self.assertTrue(np.array_equal(dec_new.units, dec.units))
    #     self.assertTrue(np.array_equal(dec_new.bounding_box[0], dec.bounding_box[0]))
    #     self.assertTrue(np.array_equal(dec_new.bounding_box[1], dec.bounding_box[1]))
    #     self.assertTrue(dec_new.states == dec.states)
    #     self.assertTrue(dec_new.states_to_bound == dec.states_to_bound)
    #     self.assertTrue(np.array_equal(dec_new.drives_neurons, dec.drives_neurons))

    #     # self.assertTrue(dec.filt == dec_new.filt)




if __name__ == '__main__':
    unittest.main()