import time
import pylink

class System(object):
    def __init__(self, address=None):
        self.tracker = pylink.EyeLink(address)
        self.edfname = "%s.edf"%time.strftime("%Y-%m-%d_%I:%M:%p")
        self.tracker.openDataFile(self.edfname)
        self.tracker.setOfflineMode()
    
    def start(self):
        self.tracker.startRecording(1,0,1,0)
        pylink.beginRealTimeMode(100)
    
    def get(self):
        self.tracker.waitForData(100, 1, 0)
        samp = self.tracker.getFloatData()
        if samp is not None:
            return samp.getLeftEye().getRawPupil()
    
    def retrieve(self, filename):
        self.tracker.setOfflineMode()
        pylink.msecDelay(.5)
        self.tracker.closeDataFile()
        self.tracker.recieveDataFile(self.edfname, filename)
    
    def sendMsg(self, msg):
        self.tracker.sendMessage(msg)

    def __del__(self):
        self.tracker.close()