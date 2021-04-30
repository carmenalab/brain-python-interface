'''
Base code for 'motiontracker' feature, compatible with PhaseSpace motiontracker
'''

import os
import time
import numpy as np

try:
    from OWL import *
except:
    OWL_MODE2 = False
    print("Cannot find phasespace driver")

cwd = os.path.split(os.path.abspath(__file__))[0]

class Simulate(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    update_freq = 240
    def __init__(self, marker_count=8, radius=(10, 2, 5), offset=(-20,0,0), speed=(5,5,4)):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.n = marker_count
        self.radius = radius
        self.offset = np.array(offset)
        self.speed = speed

        self.offsets = np.random.rand(self.n)*np.pi

    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.stime = time.time()

    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        time.sleep(1./self.update_freq)
        ts = (time.time() - self.stime)
        data = np.zeros((self.n, 3))
        for i, p in enumerate(self.offsets):
            x = self.radius[0] * np.cos(ts / self.speed[0] * 2*np.pi + p)
            y = self.radius[1] * np.sin(ts / self.speed[1] * 2*np.pi + p)
            z = self.radius[2] * np.sin(ts / self.speed[2] * 2*np.pi + p)
            data[i] = x,y,z

        #expands the dimension for HDFwriter saving
        data_temp = np.hstack([data + np.random.randn(self.n, 3) * 0.1, np.ones((self.n, 1))])
        data_temp_expand = np.expand_dims(data_temp, axis = 0)

        return data_temp_expand

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return 


class System(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    update_freq = 240
    def __init__(self, marker_count=8, server_name='10.0.0.11', init_flags=OWL_MODE2):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.marker_count = marker_count
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
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        # set markers
        for i in range(self.marker_count):
            owlMarkeri(MARKER(self.tracker, i), OWL_SET_LED, i)
        owlTracker(self.tracker, OWL_ENABLE)
        self.coords = np.zeros((self.marker_count, 4))
    
    def start(self, filename=None):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.filename = filename
        if filename is not None:
            #figure out command to tell phasespace to start a recording
            pass
        owlSetInteger(OWL_STREAMING, OWL_ENABLE)
        #owlSetInteger(OWL_INTERPOLATION, 4)

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if self.filename is not None:
            #tell phasespace to stop recording
            pass
        owlSetInteger(OWL_STREAMING, OWL_DISABLE)
    
    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        markers = []
        n = owlGetMarkers(markers, self.marker_count)
        while n == 0:
            time.sleep(.001)
            n = owlGetMarkers(markers, self.marker_count)
            
        for i, m in enumerate(markers):
            self.coords[i] = m.x, m.y, m.z, m.cond

        return self.coords

    def __del__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        for i in range(self.marker_count):
            owlMarker(MARKER(self.tracker, i), OWL_CLEAR_MARKER)   
        owlTracker(self.tracker, OWL_DESTROY)
        owlDone()

class AligningSystem(System):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def _init_markers(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        MAX = 32
        for i in range(self.marker_count):
            owlMarkeri(MARKER(self.tracker, i), OWL_SET_LED, i)
        for i in range(6):
            owlMarkeri(MARKER(self.tracker, self.marker_count+i), OWL_SET_LED, MAX+i)
        self.marker_count += 6
        owlTracker(self.tracker, OWL_ENABLE)
        self.coords = np.zeros((self.marker_count, 4))

def owl_get_error(s, n):
    """
    Print OWL error.
    Docstring

    Parameters
    ----------

    Returns
    -------
    """
    if(n < 0): return "%s: %d" % (s, n)
    elif(n == OWL_NO_ERROR): return "%s: No Error" % s
    elif(n == OWL_INVALID_VALUE): return "%s: Invalid Value" % s
    elif(n == OWL_INVALID_ENUM): return "%s: Invalid Enum" % s
    elif(n == OWL_INVALID_OPERATION): return "%s: Invalid Operation" % s
    else: return "%s: 0x%x" % (s, n)


def make(marker_count, cls=System, **kwargs):
    """This ridiculous function dynamically creates a class with a new init function
    Docstring

    Parameters
    ----------

    Returns
    -------
    """
    def init(self, **kwargs):
        super(self.__class__, self).__init__(marker_count=marker_count, **kwargs)
    
    dtype = np.dtype((np.float, (marker_count, 4)))
    if cls == AligningSystem:
        dtype = np.dtype((np.float, (marker_count+6, 4)))
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))


def make_autoalign_reference(data, filename=os.path.join(cwd, "alignment2.npz")):
    '''Creates an alignment that can be used with the autoaligner
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    from .stereo_opengl import xfm
    assert data.shape[1:] == (6, 3)
    mdata = np.median(data,0)
    cdata = mdata - mdata[0]
    rot1 = xfm.Quaternion.rotate_vecs(np.cross(cdata[2], cdata[1]), [0,1,0])
    rdata = rot1*cdata
    rot2 = xfm.Quaternion.rotate_vecs(rdata[1], [1, 0, 0])
    np.savez(filename, data=data, reference=rot2*rot1*cdata)
