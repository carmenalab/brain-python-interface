import time
from riglib import source, blackrock


from riglib.experiment import traits  # TODO -- not need, remove

channels = [5, 6, 7, 8]
s = source.DataSource(blackrock.Spikes, channels=channels)
s.start()
