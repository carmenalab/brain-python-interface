import numpy as np

def _split(data):
    msgtype = np.right_shift(np.bitwise_and(data[:,1], 7<<8), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(data[:,1], 120 << 8), 8).astype(np.uint8)
    rawdata = np.bitwise_and(data[:,1], 255)
    return np.vstack([data[:,0], msgtype, auxdata, rawdata]).T

def registrations(data):
    if data.shape[1] != 4:
        data = _split(data)

    idx = data[:,1] == 2
    sidx = data[:,1] == 3
    sysid = data[idx][:,2]
    shapeid = data[sidx][:,2]
    names = data[idx][:,3].astype(np.uint8)
    systems = dict()
    for sys in np.unique(sysid):
        name = names[sysid == sys].tostring()
        dtype = data[sidx][shapeid == sys][:,3].astype(np.uint8).tostring()
        systems[sys] = name, np.dtype(dtype)
        
    return systems

def fulldata(data):
    header = np.bitwise_and(data[:,1], 255<<8)

def rowbyte(data):
    header = np.bitwise_and(data[:,1], 255<<8)
