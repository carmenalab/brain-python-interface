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
        te = dbfn.TaskEntry(2424)
        dec_record = te.decoder_record
        dec = dec_record.load()
        
        training_block = dbfn.TaskEntry(dec_record.entry_id)
        datafiles = models.DataFile.objects.filter(entry_id=training_block.id)
        files = dict((d.system.name, d.get_path()) for d in datafiles)
        
        from riglib.bmi import extractor
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict()
        extractor_kwargs['units'] = dec.units
        extractor_kwargs['n_subbins'] = dec.n_subbins 

        # dec_new = train._train_KFDecoder_visual_feedback(cells=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
        dec_new = train._train_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
        for attr in dec.filt.model_attrs:
            old_attr = dec.filt.C
            new_attr = dec_new.filt.C
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
        te = dbfn.TaskEntry(2425)
        dec_record = te.decoder_record
        dec = dec_record.load()
        
        training_block = dbfn.TaskEntry(dec_record.entry_id)
        datafiles = models.DataFile.objects.filter(entry_id=training_block.id)
        files = dict((d.system.name, d.get_path()) for d in datafiles)
        
        from riglib.bmi import extractor
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict()
        extractor_kwargs['units'] = dec.units
        extractor_kwargs['n_subbins'] = dec.n_subbins 

        # dec_new = train._train_PPFDecoder_visual_feedback(cells=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
        dec_new = train._train_PPFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)
        for attr in dec.filt.model_attrs:
            old_attr = dec.filt.C
            new_attr = dec_new.filt.C
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
