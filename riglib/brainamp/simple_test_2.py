import time
from riglib import source, brainamp


from riglib.experiment import traits  # TODO -- not need, remove

channels = [1]
s = source.MultiChanDataSource(brainamp.EMG, channels=channels)
s.start()
