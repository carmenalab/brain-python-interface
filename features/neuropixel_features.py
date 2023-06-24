'''
Features for interacting with neuropixels
'''

import datetime
import numpy as np
from riglib.experiment import traits
import requests
import json
from riglib import experiment

# This class is almost same as open ephys python tool other than timeout option
class OpenEphysHTTPServer:
    def __init__(self, address='10.155.205.108',timeout=0.5):
        self.address = 'http://' + address + ':37497'
        self.timeout = timeout

    def send(self, endpoint, payload=None):
        # Send a request to the server
        try:
            if payload is None:
                resp = requests.get(self.address + endpoint, timeout=self.timeout)
            else:
                resp = requests.put(self.address + endpoint, data = json.dumps(payload), timeout=self.timeout)
            response = resp.json()
        except Exception as e:
            print(e, 'Check if you started openephys in neuropixel computer')
            resp = {}
            resp['mode'] = 'No connection with openephys'
            response = resp
        return response

    def set_parent_dir(self,path):
        # Set the parent directory
        payload = {'parent_directory' : path}
        data = self.send('/api/recording', payload=payload)
        return data

    def set_prepend_text(self,text):
        # Set the prepend name for the recording directory
        payload = {'prepend_text' : text}
        data = self.send('/api/recording', payload=payload)
        return data

    def set_base_text(self,text):
        # Set the base name for the recording directory
        payload = {'base_text' : text}
        data = self.send('/api/recording', payload=payload)
        return data

    def set_append_text(self,text):
        # Set the append name for the recording directory
        payload = {'append_text' : text}
        data = self.send('/api/recording', payload=payload)
        return data

    def status(self):
        # Get the current status of the GUI (IDLE, ACQUIRE, or RECORD)
        return self.send('/api/status')['mode']

    def acquire(self):
        # Start acquisition
        payload = { 'mode' : 'ACQUIRE',}
        data = self.send('/api/status', payload=payload)
        return data['mode']

    def record(self):
        # Start recording data
        previous_mode = self.status()
        payload = {'mode' : 'RECORD',}
        data = self.send('/api/status', payload=payload)       
        return data['mode']

    def idle(self):
        # Stop acquisition or recording data
        previous_mode = self.status()
        payload = { 'mode' : 'IDLE',}
        data = self.send('/api/status', payload=payload)
        return data['mode']


class RecordNeuropixels(traits.HasTraits):

    neuropixel_port1 = traits.Int(0, desc="Channel number you used for recoding with port 1 probe")
    neuropxiel_port2 = traits.Int(0, desc="Channel number hole you used for recoding with port 2 probe")
    neuropixel_mapping_file = traits.String("", desc="Name of channel mapping file")

    IP_neuropixel = '10.155.205.108'
    gui = OpenEphysHTTPServer(IP_neuropixel, timeout=0.5)
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
    def pre_init(cls, saveid=None, subject_name=None, gui=gui,**kwargs):
        cls.openephys_status = 'IDLE'
        prepend_text = str(datetime.date.today())
        filename = f'_Neuropixel_{subject_name}_te'
        append_text = str(saveid) if saveid else 'Test'

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
