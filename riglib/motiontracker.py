import time
import numpy as np
try:
    from OWL import *
except:
    OWL_MODE2 = False
    print "Cannot find phasespace driver"

class Simulate(object):
    update_freq = 240
    def __init__(self, marker_count=8, radius=(10, 2, 5), offset=(-20,0,0), speed=(5,5,4)):
        self.n = marker_count
        self.radius = radius
        self.offset = np.array(offset)
        self.speed = speed

        self.offsets = np.random.rand(self.n)*np.pi

    def start(self):
        self.stime = time.time()

    def get(self):
        time.sleep(1./self.update_freq)
        ts = (time.time() - self.stime)
        data = np.zeros((self.n, 3))
        for i, p in enumerate(self.offsets):
            x = self.radius[0] * np.cos(ts / self.speed[0] * 2*np.pi + p)
            y = self.radius[1] * np.sin(ts / self.speed[1] * 2*np.pi + p)
            z = self.radius[2] * np.sin(ts / self.speed[2] * 2*np.pi + p)
            data[i] = x,y,z

        return data + np.random.randn(self.n, 3)*0.1

    def stop(self):
        return 

    def testfunc(self):
        return "blah"


class System(object):
    update_freq = 240
    def __init__(self, marker_count=8, server_name='10.0.0.11', init_flags=OWL_MODE2):
        self.marker_count = marker_count
        self.coords = np.zeros((self.marker_count, 4))
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
        #owlSetInteger(OWL_INTERPOLATION, 4)

    def stop(self):
        if self.filename is not None:

            #tell phasespace to stop recording
            pass
        owlSetInteger(OWL_STREAMING, OWL_DISABLE)
    
    def get(self):
        markers = []
        
        n = owlGetMarkers(markers, self.marker_count)
        while n == 0:
            time.sleep(.001)
            n = owlGetMarkers(markers, self.marker_count)
            
        for i, m in enumerate(markers):
            self.coords[i] = m.x, m.y, m.z, m.cond

        return self.coords

    def __del__(self):
        for i in range(self.marker_count):
            owlMarker(MARKER(self.tracker, i), OWL_CLEAR_MARKER)   
        owlTracker(self.tracker, OWL_DESTROY)
        owlDone()

def owl_get_error(s, n):
    """Print OWL error."""
    if(n < 0): return "%s: %d" % (s, n)
    elif(n == OWL_NO_ERROR): return "%s: No Error" % s
    elif(n == OWL_INVALID_VALUE): return "%s: Invalid Value" % s
    elif(n == OWL_INVALID_ENUM): return "%s: Invalid Enum" % s
    elif(n == OWL_INVALID_OPERATION): return "%s: Invalid Operation" % s
    else: return "%s: 0x%x" % (s, n)


def make_system(marker_count, **kwargs):
    """This ridiculous function dynamically creates a class with a new init function"""
    def init(self, **kwargs):
        super(self.__class__, self).__init__(marker_count=marker_count, **kwargs)

    dtype = np.dtype((np.float, (marker_count, 4)))
    return type("System", (System,), dict(dtype=dtype, __init__=init))

def make_simulate(marker_count, **kwargs):
    def init(self, **kwargs):
        super(self.__class__, self).__init__(marker_count=marker_count, **kwargs)

    dtype = np.dtype((np.float, (marker_count, 3)))
    return type("Simulate", (Simulate,), dict(dtype=dtype, __init__=init))    
