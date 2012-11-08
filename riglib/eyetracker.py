import time
import itertools
import numpy as np
try:
    import pylink
except ImportError:
    print "Couldn't find eyetracker module"

class Simulate(object):
    update_freq = 500
    dtype = np.dtype((np.float, (2,)))

    def __init__(self, fixations=[(0,0), (-0.6,0.3), (0.6,0.3)], isi=500, slen=15):
        from scipy.interpolate import interp1d
        flen = range(len(fixations)+1)
        t = list(itertools.chain(*[(i*isi + slen*i, (i+1)*isi + slen*i) for i in flen]))[:-1]
        xy = np.append(np.tile(fixations, (1, 2)).reshape(-1, 2), [fixations[0]], axis=0)
        self.mod = t[-1] / 1000.
        self.interp = interp1d(np.array(t)/1000., xy, kind='linear', axis=0)
        self.fixations = fixations
        self.isi = isi

    def start(self):
        self.stime = time.time()

    def get(self):
        time.sleep(1./update_freq)
        return self.interp((time.time() - self.stime) % self.mod) + np.random.randn(2)*.01

    def stop(self):
        return 

class System(object):
    update_freq = 500
    dtype = np.dtype((np.float, (2,)))

    def __init__(self, address='10.0.0.2'):
        self.tracker = pylink.EyeLink(address)
        self.tracker.setOfflineMode()
    
    def start(self, filename=None):
        self.filename = filename
        if filename is None:
            self.filename = "%s.edf"%time.strftime("%Y%m%d") #%Y-%m-%d_%I:%M:%p
        self.tracker.openDataFile(self.filename)
        pylink.beginRealTimeMode(100)
        self.tracker.startRecording(1,0,1,0)

    def stop(self):
        self.tracker.stopRecording()
        pylink.endRealTimeMode()
    
    def get(self):
        samp = self.tracker.getNextData()
        while samp != pylink.SAMPLE_TYPE:
            time.sleep(.001)
            samp = self.tracker.getNextData()
        try:
            data = np.array(self.tracker.getFloatData().getLeftEye().getGaze())
            if data.sum() < -1e4:
                return np.array([np.nan, np.nan])
        except:
            return np.array([np.nan, np.nan])
            
        return data
        
    def set_filter(self, filt):
        self.filter = filt
    
    def retrieve(self, filename):
        self.tracker.setOfflineMode()
        pylink.msecDelay(1)
        self.tracker.closeDataFile()
        self.tracker.receiveDataFile(self.filename, filename)
    
    def sendMsg(self, msg):
        self.tracker.sendMessage(msg)

    def __del__(self):
        self.tracker.close()
