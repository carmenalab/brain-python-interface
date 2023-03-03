'''
Features for interacting with neuropixels
'''
import time
import os
import numpy as np
from open_ephys.control import OpenEphysHTTPServer



#gui.acquire(10)

#gui.record(10)

#print(gui.status())

class RecordNeuropixels(traits.HasTraits):

    self.IP_neuropixel = '10.155.205.108'
    self.gui = OpenEphysHTTPServer(self.IP_neuropixel)

    def cleanup(self, database, saveid, **kwargs):

    @classmethod
    def pre_init(cls, saveid=None, record_headstage=False, headstage_connector=None, headstage_channels=None, **kwargs):
        gui.record()