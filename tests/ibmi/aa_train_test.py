from riglib.bmi import train, extractor
import numpy as np
from db.tracker import models

channels = range(1, 7)
units = np.hstack([np.array(channels).reshape(-1,1), np.ones((len(channels),1), dtype=np.int)])
extractor_cls = extractor.BinnedSpikeCountsExtractor
extractor_kwargs = dict()
extractor_kwargs['units'] = units
extractor_kwargs['n_subbins'] = 1

armassist_state_space = train.armassist_state_space

entry = 10
datafiles = models.DataFile.objects.filter(entry_id=entry)
 
# this is sort of a hack, fix later
# old inputdata dict assumed there was only one datafile associated with 
# each system, but this is not always the case (e.g., Blackrock has both 
# nev and nsx files) -- in this case, set the corresponding dict value as
# a list of files
# inputdata = dict((d.system.name, d.get_path()) for d in datafiles)
inputdata = dict()
system_names = set(d.system.name for d in datafiles)
for system_name in system_names:
    files = [d.get_path() for d in datafiles if d.system.name == system_name]
    if system_name == 'blackrock':
        inputdata[system_name] = files  # list of (one or more) files
    else:
        assert(len(files) == 1)
        inputdata[system_name] = files[0]  # just one file

files = inputdata
dec = train._train_armassist_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=None, binlen=0.1, tslice=[None,None],
    _ssm=armassist_state_space, source='task', kin_var=None, shuffle=False, **files)
