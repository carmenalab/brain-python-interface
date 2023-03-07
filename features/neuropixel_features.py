'''
Features for interacting with neuropixels
'''
import time
import os
import numpy as np
from riglib.experiment import traits
from open_ephys.control import OpenEphysHTTPServer

class RecordNeuropixels(traits.HasTraits):

    IP_neuropixel = '10.155.205.108'
    gui = OpenEphysHTTPServer(IP_neuropixel)

    def cleanup(self,database, saveid, gui=gui, **kwargs):
        #super().cleanup(database, saveid, gui=gui, **kwargs)
        try:
            gui.acquire()
        except Exception as e:
            print(e)
            print('\n\ncould not stop OpenEphys recording. Please manually stop the recording\n\n')
        
    @classmethod
    def pre_init(cls, saveid=None, gui=gui,**kwargs):
        if saveid is not None:
            try:
                gui.record()
            except Exception as e:
                print(e)
                print('\n\ncould not start OpenEphys recording\n\n')
        else:
            try:
                gui.acquire()
            except Exception as e:
                print(e)
                print('\n\ncould not start OpenEphys acquisition\n\n')
        print(f'Open Ephys status : {gui.status()}')
  
        #if hasattr(super(), 'pre_init'):
        #    super().pre_init(saveid=saveid,gui=gui,**kwargs)

    # def run(self):
    #     super().run()
    #     time.sleep(0.1) # Wait a bit to be sure the recording started
    #     IP_neuropixel = '10.155.205.108'
    #     gui = OpenEphysHTTPServer(IP_neuropixel)
    #     print(f'Open Ephys status : {gui.status()}')