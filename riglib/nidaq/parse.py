'''Needs docs'''


import numpy as np

msgtype_mask = 0b0000111<<8
auxdata_mask = 0b1111000<<8
rawdata_mask = 0b11111111

def _split(data, flip=False):
    '''
    '''
    # If the data is a 1D array, extract the timestamps and the raw event codes
    if len(data.shape) < 2:
        data = np.array(data[data['chan'] == 257][['ts', 'unit']].tolist())
    msgs = data[:,1].astype(np.int16)
    
    if not flip:
        msgs = ~msgs # bit-flip the binary messages
    msgtype = np.right_shift(np.bitwise_and(msgs, msgtype_mask), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8).astype(np.uint8)
    rawdata = np.bitwise_and(msgs, rawdata_mask)
    return np.vstack([data[:,0], msgtype, auxdata, rawdata]).T

def registrations(data):
    if data.ndim < 2 or data.shape[1] != 4:
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
        systems[sys] = name[:-1], dtype[:-1]
        
    return systems

def rowbyte(data, **kwargs):
    if data.ndim < 2 or data.shape[1] != 4:
        data = _split(data, **kwargs)
    #reg = registrations(data)

    msgs = data[data[:,1] == 5]
    systems = dict()
    for i in np.unique(msgs[:,2]):
        systems[i] = msgs[msgs[:,2] == i][:,[0,-1]]
    return systems

def messages(data, **kwargs):
    if data.ndim < 2 or data.shape[1] != 4:
        data = _split(data, **kwargs)

    times = data[data[:,1] == 1, 0]
    names = data[data[:,1] == 1,-1].astype(np.uint8)

    tidx, = np.nonzero(names == 0)
    tidx = np.hstack([0, tidx+1])

    msgs = []
    for s, e in np.vstack([tidx[:-1], tidx[1:]-1]).T:
        msgs.append((times[s], names[s:e].tostring()))

    return np.array(msgs, dtype=[('time', np.float), ('state', 'S256')])
