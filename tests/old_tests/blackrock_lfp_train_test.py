import numpy as np
from db import dbfunctions as dbfn
from db.tracker import models
from riglib.bmi import train, kfdecoder, ppfdecoder
import unittest

datafiles = models.DataFile.objects.filter(entry_id=3)

inputdata = dict()
system_names = set(d.system.name for d in datafiles)
for system_name in system_names:
    files = [d.get_path() for d in datafiles if d.system.name == system_name]
    if system_name == 'blackrock':
        inputdata[system_name] = files  # list of (one or more) files
    else:
        assert(len(files) == 1)
        inputdata[system_name] = files[0]  # just one file


from riglib.bmi import extractor


channels = list(range(1, 7))
units = np.hstack([np.array(channels).reshape(-1,1), np.ones((len(channels),1), dtype=np.int)])
extractor_cls = extractor.LFPButterBPFPowerExtractor
extractor_kwargs = dict()
extractor_kwargs['channels'] = np.unique(units[:,0])

# extractor_cls = extractor.BinnedSpikeCountsExtractor
# units = np.array([[1, 1], [3, 1], [5, 1], [8, 1]])
# extractor_kwargs = dict()
# extractor_kwargs['units'] = units
# extractor_kwargs['n_subbins'] = 1

dec_new = train._train_ismore_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=units, binlen=0.1, tslice=[4, 54], **inputdata)
