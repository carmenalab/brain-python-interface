#!/usr/bin/python
'''
Equivalence test for changes to vfb training procedure (riglib.bmi.train._train_KFDecoder_visual_feedback)
'''
import numpy as np
from db import dbfunctions as dbfn
from db.tracker import models
from riglib.bmi import train
reload(train)

te = dbfn.get_task_entry(1844)
dec_record = dbfn.get_decoder_entry(te)
dec = dbfn.get_decoder(te)

training_block = dbfn.get_task_entry(dec_record.entry_id)
datafiles = models.DataFile.objects.filter(entry_id=training_block.id)
files = dict((d.system.name, d.get_path()) for d in datafiles)

dec_new = train._train_KFDecoder_visual_feedback(cells=dec.units, binlen=dec.binlen, tslice=dec.tslice, **files)

print("C error: %g" % np.max(np.abs(dec_new.kf.C - dec_new.kf.C)))
print("Q error: %g" % np.max(np.abs(dec_new.kf.Q - dec_new.kf.Q)))
print("R error: %g" % np.max(np.abs(dec_new.kf.R - dec_new.kf.R)))
print("S error: %g" % np.max(np.abs(dec_new.kf.S - dec_new.kf.S)))
print("T error: %g" % np.max(np.abs(dec_new.kf.T - dec_new.kf.T)))
print("mFR error: %g" % np.max(np.abs(dec_new.mFR - dec.mFR)))

dec_new = train._train_PPFDecoder_visual_feedback(cells=dec.units, tslice=dec.tslice, **files)
