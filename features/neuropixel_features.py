'''
Features for interacting with neuropixels
'''

import datetime
import numpy as np
from riglib.experiment import traits
from riglib import experiment
from riglib.open_ephys import OpenEphysHTTPServer

class RecordNeuropixels(traits.HasTraits):

    neuropixel_port1_chamber = traits.String("", desc="the channel position (e.g. LM1)")
    neuropixel_port1_drive_type = traits.String("", desc="the name of the hole layout")
    neuropixel_port1_site = traits.Int(0, desc="the hole number of the penetration")
    neuropixel_port1_depth = traits.Float(0, desc="the recording depth in um")
    neuropxiel_port2_chamber = traits.String("", desc="the channel position (e.g. LM1)")
    neuropixel_port2_drive_type = traits.String("", desc="the name of the hole layout")
    neuropixel_port2_site = traits.Int(0, desc="the hole number of the penetration")
    neuropixel_port2_depth = traits.Float(0, desc="the recording depth in um")
    gui = OpenEphysHTTPServer('10.155.205.108', timeout=0.5)
    
    def cleanup(self,database, saveid, gui=gui, **kwargs):
        super().cleanup(database, saveid, gui=gui, **kwargs)
        try:
            gui.acquire()
        except Exception as e:
            print(e)
            print('\n\ncould not stop OpenEphys recording. Please manually stop the recording\n\n')
        
    @classmethod
    def pre_init(cls, saveid=None, subject_name=None, gui=gui,**kwargs):
        cls.openephys_status = 'IDLE'
        prepend_text = str(datetime.date.today())
        filename = f'_Neuropixel_{subject_name}_te'
        append_text = str(saveid) if saveid else 'Test'
        parent_dir = 'E://Neuropixel_data'

        gui.set_parent_dir(parent_dir)
        gui.set_prepend_text(prepend_text)
        gui.set_base_text(filename)
        gui.set_append_text(append_text)

        if saveid is not None:
            try:
                gui.record()
                cls.openephys_status = gui.status()
            except Exception as e:
                print(e)
                print('\n\ncould not start OpenEphys recording\n\n')
        else:
            try:
                gui.acquire()
                cls.openephys_status = gui.status()
            except Exception as e:
                print(e)
                print('\n\ncould not start OpenEphys acquisition\n\n')
        print(f'Open Ephys status : {gui.status()}')
  
        if hasattr(super(), 'pre_init'):
            super().pre_init(saveid=saveid,gui=gui,**kwargs)

    def run(self):
        if not self.openephys_status in ["ACQUIRE", "RECORD"]:
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.openephys_status)
            self.termination_err.seek(0)
            self.state = None
        try:
            super().run()
        except Exception as e:
            print(e)
