from ..riglib import source
from ..riglib.brainamp import rda

channels = [1]
s = source.MultiChanDataSource(rda.EMGData, channels=channels, recorder_ip='192.168.137.1')
s.start()
