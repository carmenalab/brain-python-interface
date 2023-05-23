#import os
import datetime
#import numpy as np
#from open_ephys.control import OpenEphysHTTPServer
import requests
import json

IP_neuropixel = '10.155.205.108'
address = f'http://{IP_neuropixel}:37497'

# These functions are almost same as open ephys python tool other than time-out option
class OpenEphysHTTPServer:
    def __init__(self, address='10.155.205.108',timeout=0.5):
        self.address = 'http://' + address + ':37497'
        self.timeout = timeout

    def send(self, endpoint, payload=None):
        # Send a request to the server
            if payload is None:
                resp = requests.get(self.address + endpoint, timeout=self.timeout)
            else:
                resp = requests.put(self.address + endpoint, data = json.dumps(payload), timeout=self.timeout)
            return resp.json()

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

\
gui = OpenEphysHTTPServer(IP_neuropixel)
parent_dir = 'E://Neuropixel_data'

prepend_text = str(datetime.date.today())
filename = '_Neuropixel_'
append_text = 'test6'

gui.set_prepend_text(prepend_text)
gui.set_base_text(filename)
gui.set_append_text(append_text)
gui.acquire()

#gui.set_start_new_dir() # This isn't necessary because saved directory changes when file name changes
#gui.record(10)

# gui.set_parent_dir(parent_dir)
# print(gui.get_recording_info())
#print(gui.get_processors('Neuropix-PXI'))
#A = gui.get_parameters(105,0)
#print(A)
#print(gui.get_parameters(103,1))