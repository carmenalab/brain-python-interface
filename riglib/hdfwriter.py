import tables
import shm

class MsgTable(tables.IsDescription):
    time = tables.UIntCol()
    msg = tables.StringCol(256)

class HDFWriter(object):
    def __init__(self, filename, systems=None):
        print "Saving datafile to %s"%filename
        self.h5 = tables.openFile(filename, "w")
        self.filter = tables.Filters(complevel=5, complib="zlib", shuffle=True)
        self.data = {}
        self.msgs = {}
        
        for s in systems:
            self.register(s)
    
    def register(self, system):
        if isinstance(system, shm.DataSource):
            dataname = system.source.__module__.split('.')[-1]
            source = system.source
        else:
            dataname = system.__module__.split('.')[-1]
            source = system
        
        arr = self.h5.createEArray("/", dataname, tables.FloatAtom(), 
            shape=(0,)+system.slice_shape, expectedrows=system.bufferlen*100,
            filters=self.filter)
        msg = self.h5.createTable("/", dataname+"_msgs", MsgTable,
            expectedrows=system.bufferlen*100, filters=self.filter)

        self.data[source] = arr
        self.msgs[source] = msg
    
    def send(self, system, data):
        self.data[system].append([data])

    def sendMsg(self, system, msg):
        row = self.msgs[system].row
        row['time'] = len(self.data[system])
        row['msg'] = msg
        row.append()
    
    def __del__(self):
        self.h5.close()