import numpy as np

def _split(data):
    msgtype = np.right_shift(np.bitwise_and(data[:,1], 0b0000111<<8), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(data[:,1], 0b1111000<<8), 8).astype(np.uint8)
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

def rowbyte(data):
    if data.shape[1] != 4:
        data = _split(data)
    reg = registrations(data)

    msgs = data[data[:,1] == 5]
    systems = dict()
    for sys, (name, dtype) in reg.items():
        systems[name] = msgs[msgs[:,2] == sys, [0,-1]]
    return systems

def messages(data, thres=1000):
    if data.shape[1] != 4:
        data = _split(data)

    times = msgs[data[:,1] == 1, 0]
    names = msgs[data[:,1] == 1,-1].astype(np.uint8)

    tidx, = np.nonzero(np.diff(times) > 1000)
    tidx = np.hstack([times[0], tidx+1, times[-1]])

    msgs = []
    for s, e in np.vstack([tidx[:-1], tidx[1:]]).T:
        msgs.append((times[s], names[s:e].tostring()))
    return msgs
