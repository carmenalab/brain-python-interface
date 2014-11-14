import time
from riglib import source, blackrock

channels = [5, 6, 7, 8]
s = source.DataSource(blackrock.Spikes, channels=channels)
s.start()
