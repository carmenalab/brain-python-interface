import tables

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
        self.indices = {}
        
        for s in systems:
            self.register(s)
    
    def register(self, system):
        dataname = system.__module__.split('.')[0]
        print "registered %s"%dataname

        arr = self.h5.createEArray("/", dataname, tables.FloatAtom(), 
            shape=(0,)+system.slice_shape, expectedrows=system.bufferlen*100,
            filters=self.filter)
        msg = self.h5.createTable("/", dataname+"_msgs", MsgTable,
            expectedrows=system.bufferlen*100, filters=self.filter)
        self.data[system] = arr
        self.msgs[system] = msg
        self.indices[system] = 0
    
    def send(self, system, data):
        print "Wrote into %s"%system
        self.data[system].append(data)
        self.indices[system] += 1
    
    def sendMsg(self, system, msg):
        row = self.msgs[system].row
        row['time'] = self.indices[system]
        row['msg'] = msg
        row.append()
    
    def __del__(self):
        self.h5.close()