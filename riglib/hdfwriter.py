import tables
import shm

compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)

class MsgTable(tables.IsDescription):
    time = tables.UIntCol()
    msg = tables.StringCol(256)

class HDFWriter(object):
    def __init__(self, filename):
        print "Saving datafile to %s"%filename
        self.h5 = tables.openFile(filename, "w")
        self.data = {}
        self.msgs = {}
    
    def register(self, name, dtype):
        print "HDFwriter registered %r"%name
        if dtype.subdtype is not None:
            #just a simple dtype with a shape
            dtype, sliceshape = dtype.subdtype
            arr = self.h5.createEArray("/", name, tables.Atom.from_dtype(dtype), 
                shape=(0,)+sliceshape, filters=compfilt)
        else:
            arr = self.h5.createTable("/", name, dtype, filters=compfilt)

        msg = self.h5.createTable("/", name+"_msgs", MsgTable, filters=compfilt)

        self.data[name] = arr
        self.msgs[name] = msg
    
    def send(self, system, data):
        self.data[system].append([data])

    def sendMsg(self, system, msg):
        row = self.msgs[system].row
        row['time'] = len(self.data[system])
        row['msg'] = msg
        row.append()
    
    def close(self):
        print "Closed hdf"
        self.h5.close()
