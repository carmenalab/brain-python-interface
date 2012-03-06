import time
import numpy as np
from OWL import *

class System(object):
    def __init__(self, marker_count=1,server_name='10.0.0.11', init_flags=0):
        if(owlInit(server_name, init_flags) < 0):
            raise Exception(owl_get_error("init error",owlGetError()))
                
        # flush requests and check for errors fix
        if(owlGetStatus() == 0):
            raise Exception(owl_get_error("error in point tracker setup", owlGetError()))
        
        # set define frequency
        owlSetFloat(OWL_FREQUENCY, OWL_MAX_FREQUENCY)
    
        #create a point tracker
        self.tracker = 0
        owlTrackeri(self.tracker, OWL_CREATE, OWL_POINT_TRACKER)
    
        # set markers
        for i in range(marker_count):
            owlMarkeri(MARKER(self.tracker, i), OWL_SET_LED, i)
        owlTracker(self.tracker, OWL_ENABLE)
    
    def start(self, filename=None):
        self.filename = filename
        if filename is not None:
            #figure out command to tell phasespace to start a recording
            pass
        owlSetInteger(OWL_STREAMING, OWL_ENABLE)

    def stop(self):
        if self.filename is not None:
            #tell phasespace to stop recording
            pass
        owlSetInteger(OWL_STREAMING, OWL_DISABLE)
    
    def get(self):
        markers=[]
        coords = np.zeros((32, 3))
        n = owlGetMarkers(markers, 32)
        for i in range(n):
            if markers[i].cond > 0:
                coords[i] = markers[i].x, markers[i].y, markers[i].z
        return coords
        
    def retrieve(self, filename):
        pass
    
    def sendMsg(self, msg):
        pass

    def __del__(self):
        for i in range(marker_count):
            owlMarker(MARKER(self.tracker, i), OWL_CLEAR_MARKER)   
        self.owlTracker(self.tracker, OWL_DESTROY)
        owlDone()

def owl_get_error(s, n):
    """Print OWL error."""
    if(n < 0): return "%s: %d" % (s, n)
    elif(n == OWL_NO_ERROR): return "%s: No Error" % s
    elif(n == OWL_INVALID_VALUE): return "%s: Invalid Value" % s
    elif(n == OWL_INVALID_ENUM): return "%s: Invalid Enum" % s
    elif(n == OWL_INVALID_OPERATION): return "%s: Invalid Operation" % s
    else: return "%s: 0x%x" % (s, n)
