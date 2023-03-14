'''
Features for interacting with neuropixels
'''
import time
import os
import datetime
import numpy as np
from riglib.experiment import traits
from open_ephys.control import OpenEphysHTTPServer

class RecordNeuropixels(traits.HasTraits):

    IP_neuropixel = '10.155.205.108'
    gui = OpenEphysHTTPServer(IP_neuropixel)
    parent_dir = 'E://Neuropixel_data'
    gui.set_parent_dir(parent_dir)

    def cleanup(self,database, saveid, gui=gui, **kwargs):
        super().cleanup(database, saveid, gui=gui, **kwargs)
        try:
            gui.acquire()
        except Exception as e:
            print(e)
            print('\n\ncould not stop OpenEphys recording. Please manually stop the recording\n\n')
        
    @classmethod
    def pre_init(cls, saveid=None, gui=gui,**kwargs):
        prepend_text = str(datetime.date.today())
        filename = '_Neuropixel_te'
        append_text = str(saveid) if saveid else 'Test'
        gui.set_prepend_text(prepend_text)
        gui.set_base_text(filename)
        gui.set_append_text(append_text)
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
  
        if hasattr(super(), 'pre_init'):
            super().pre_init(saveid=saveid,gui=gui,**kwargs)
