import time
import pylink

class System(object):
    def __init__(self, address='10.0.0.2'):
        self.tracker = pylink.EyeLink(address)
        self.tracker.setOfflineMode()
    
    def start(self, filename=None):
        self.filename = filename
        if filename is None:
            self.filename = "%s.edf"%time.strftime("%Y%m%d") #%Y-%m-%d_%I:%M:%p
        self.tracker.openDataFile(self.filename)
        self.tracker.startRecording(1,0,1,0)
        #pylink.beginRealTimeMode(100)

    def stop(self):
        self.tracker.stopRecording()
        pylink.endRealTimeMode()
    
    def get(self):
        samp = self.tracker.getNextData()
        while samp != pylink.SAMPLE_TYPE:
            time.sleep(1/750.)
            samp = self.tracker.getNextData()
        
        return self.tracker.getFloatData().getRightEye().getGaze()
    
    def retrieve(self, filename):
        self.tracker.setOfflineMode()
        pylink.msecDelay(.5)
        self.tracker.closeDataFile()
        self.tracker.recieveDataFile(self.edfname, filename)
    
    def sendMsg(self, msg):
        self.tracker.sendMessage(msg)

    def __del__(self):
        self.tracker.close()
