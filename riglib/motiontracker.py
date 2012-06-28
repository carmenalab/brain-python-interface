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

        return np.hstack([data + np.random.randn(self.n, 3)*0.1, np.ones((self.n,1))])

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
        self._init_markers()

    def _init_markers(self):
        # set markers
        for i in range(self.marker_count):
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

class AligningSystem(System):
    def _init_markers(self):
        MAX = 42
        for i in range(self.marker_count):
            owlMarkeri(MARKER(self.tracker, i), OWL_SET_LED, i)
        for i in range(6):
            owlMarkeri(MARKER(self.tracker, self.marker_count+i), OWL_SET_LED, MAX+i)
        self.marker_count += 6
        owlTracker(self.tracker, OWL_ENABLE)

def owl_get_error(s, n):
    """Print OWL error."""
    if(n < 0): return "%s: %d" % (s, n)
    elif(n == OWL_NO_ERROR): return "%s: No Error" % s
    elif(n == OWL_INVALID_VALUE): return "%s: Invalid Value" % s
    elif(n == OWL_INVALID_ENUM): return "%s: Invalid Enum" % s
    elif(n == OWL_INVALID_OPERATION): return "%s: Invalid Operation" % s
    else: return "%s: 0x%x" % (s, n)


def make(marker_count, cls=System, **kwargs):
    """This ridiculous function dynamically creates a class with a new init function"""
    def init(self, **kwargs):
        super(self.__class__, self).__init__(marker_count=marker_count, **kwargs)

    if cls == AligningSystem:
        marker_count += 6
    dtype = np.dtype((np.float, (marker_count, 4)))
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))
    

from riglib.stereo_opengl import xfm
class AutoAlign(object):
    '''Runs the autoalignment filter to center everything into the chair coordinates'''
    def __init__(self, reference="alignment.npz"):
        self.ref = np.load(reference)['reference']
        self.xfm = xfm.Quaternion()
        self.off = np.array([0,0,0])
        self.abc_cross = np.cross(self.ref[2] - self.ref[0], self.ref[1] - self.ref[0])
        self.def_cross = np.cross(self.ref[5] - self.ref[3], self.ref[4] - self.ref[3])

    def __call__(self, data):
        mdata = data.mean(0)
        avail = ~np.isnan(mdata[-6:]).any(1)
        if avail[:3].all():
            #ABC reference
            cdata = mdata[-6:-3] - self.ref[0]
            vec = np.cross(cdata[2] - cdata[0], cdata[1] - cdata[0])
            self.off = self.ref[0]
            self.xfm = xfm.Quaternion.rotate_vecs(vec, self.abc_cross)
        elif avail[3:].all():
            #DEF reference
            cdata = mdata[-3:] - self.ref[3]
            vec = np.cross(cdata[2] - cdata[0], cdata[1] - cdata[0])
            self.off = self.ref[3]
            self.xfm = xfm.Quaternion.rotate_vecs(vec, self.def_cross)

        return self.xfm*(mdata[:-6] - self.off) + self.off


def make_autoalign_reference(data, filename="alignment.npz"):
    '''Creates an alignment that can be used with the autoaligner'''
    assert data.shape[1:] == (6, 3)
    mdata = data.mean(0)
    cdata = mdata - mdata[0]
    rot = xfm.Quaternion.rotate_vecs(np.cross(cdata[2], cdata[1]), [0,1,0])
    np.savez_compressed(filename, data=data, reference=rot*cdata)