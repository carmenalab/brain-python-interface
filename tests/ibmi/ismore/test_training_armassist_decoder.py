from riglib.bmi import train, extractor
import numpy as np
from db.tracker import models


TRAIN_SPIKE = True

entry = 284
ssm = train.armassist_state_space
binlen = 0.1
tslice = [9,69]
pos_key = 'plant_pos'


if TRAIN_SPIKE:
    channels = list(range(1, 7))
    units = np.hstack([np.array(channels).reshape(-1,1), np.ones((len(channels),1), dtype=np.int)])
    extractor_cls = extractor.BinnedSpikeCountsExtractor
    extractor_kwargs = dict()
    extractor_kwargs['units'] = units
    extractor_kwargs['n_subbins'] = 1
else:  # train an LFP decoder
    channels = np.array(list(range(1, 9)))
    units = np.hstack([channels.reshape(-1, 1), np.zeros(channels.reshape(-1, 1).shape, dtype=np.int32)])
    extractor_cls = extractor.LFPButterBPFPowerExtractor
    extractor_kwargs = dict()
    extractor_kwargs['channels'] = channels

# list of DataFile objects
datafiles = models.DataFile.objects.filter(entry_id=entry)

# key: a string representing a system name (e.g., 'plexon', 'blackrock', 'task', 'hdf')
# value: a single filename, or a list of filenames if there are more than one for that system
files = dict()
system_names = set(d.system.name for d in datafiles)
for system_name in system_names:
    filenames = [d.get_path() for d in datafiles if d.system.name == system_name]
    if system_name == 'blackrock':
        files[system_name] = filenames  # list of (one or more) files
    else:
        assert(len(filenames) == 1)
        files[system_name] = filenames[0]  # just one file

dec = train.train_KFDecoder(files, extractor_cls, extractor_kwargs, train.get_plant_pos_vel, ssm, units, update_rate=binlen, tslice=tslice, pos_key=pos_key)
