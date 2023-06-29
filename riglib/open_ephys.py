import requests
import json

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