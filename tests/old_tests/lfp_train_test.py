import numpy as np
from db import dbfunctions as dbfn
from db.tracker import models
from riglib.bmi import train, kfdecoder, ppfdecoder
import unittest

datafiles = models.DataFile.objects.filter(entry_id=3911)
files = dict((d.system.name, d.get_path()) for d in datafiles)

from riglib.bmi import extractor


channels = list(range(1, 13))
units = np.hstack([np.array(channels).reshape(-1,1), np.ones((len(channels),1), dtype=np.int)])
extractor_cls = extractor.LFPMTMPowerExtractor
extractor_kwargs = dict()
extractor_kwargs['channels'] = np.unique(units[:,0])

# extractor_cls = extractor.BinnedSpikeCountsExtractor
# units = np.array([[1, 1], [3, 1], [5, 1], [8, 1]])
# extractor_kwargs = dict()
# extractor_kwargs['units'] = units
# extractor_kwargs['n_subbins'] = 1

print('extractor:', extractor_cls)

dec_new = train._train_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=units, binlen=0.1, tslice=[2, 30], **files)
