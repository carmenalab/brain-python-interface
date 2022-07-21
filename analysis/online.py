import aopy
import numpy as np
from riglib import source
from riglib.ecube import Broadband, LFP, Digital
from riglib.bmi import state_space_models, train, extractor
import time

class DataOnline():

    def __init__(self, datasource, channels, buffer_len=5):
        self.channels = channels
        self.ds = source.MultiChanDataSource(datasource, channels=channels, buffer_len=buffer_len)
        self.ds.start()

    def get_new(self):
        return self.ds.get_new(self.channels)

    def __del__(self):
        self.ds.stop()

