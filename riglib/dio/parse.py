'''
Parse digital data from neural recording system into task data/messages/synchronization pulses
'''

import numpy as np

msgtype_mask = 0b0000111 << 8
auxdata_mask = 0b1111000 << 8
rawdata_mask = 0b11111111

MSG_TYPE_DATA = 0
MSG_TYPE_MESSAGE = 1
MSG_TYPE_REGISTER = 2
MSG_TYPE_REGISTER_SHAPE = 3
MSG_TYPE_ROW = 4
MSG_TYPE_ROWBYTE = 5

def parse_data(strobe_data):
    '''
    Parse out 'strobe' digital data into header/registrations + actual data
    '''
    reg_parsed_data = registrations(strobe_data)
    msg_parsed_data = messages(strobe_data)
    rowbyte_parsed_data = rowbyte(strobe_data)

    data = dict(messages=msg_parsed_data)
    for key in rowbyte_parsed_data:
        if key in reg_parsed_data:
            sys_name = reg_parsed_data[key][0]
            try:
                sys_dtype = np.dtype(eval(reg_parsed_data[key][1]))
            except:
                sys_dtype = reg_parsed_data[key][1]
            data[sys_name] = dict(row_ts=rowbyte_parsed_data[key])
        else:
            data[key] = dict(row_ts=rowbyte_parsed_data[key])

    return data

def _split(data, flip=False):
    '''
    Helper function to take the 16-bit integer saved in the neural data file 
    and map it back to the three fields of the message type (see docs on 
    communication protocol for details)

    Parameters
    ----------
    data : np.ndarray 
        Integer data and timestamps as stored in the neural data file when messages were sent during experiment

    Returns
    -------
    np.ndarray
        Raw message data split into the fields (type, aux, "payload")
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

def registrations(data, map_system=False):
    '''
    Parse the DIO data from the neural recording system to determine which 
    data sources were registered by the experiment software
    
    Parameters
    ---------
    data: np.array
        Digital bit data sent to the neural recording box. This data can either
        be a 1D record array with fields ('chan', 'unit', 'ts') or an N x 4
        regular array where the four columns are (timestamp, message type,
        auxiliary data, raw data)
    
    Returns
    -------
    systems : dict
        In the dictionary, keys are the ID # of the system (assigned sequentially
        during registration time when the task is initializing). Values are
        tuples of (name, dtype)
    '''
    if data.ndim < 2 or data.shape[1] != 4:
        if map_system:
            data = _split(data, flip=True)
        else:
            data = _split(data)

    ts, msgtype, auxdata, rawdata = data[:,0], data[:,1], data[:,2], data[:,3].astype(np.uint8)
    idx = msgtype == MSG_TYPE_REGISTER #data[:,1] == MSG_TYPE_REGISTER 
    shape_idx = msgtype == MSG_TYPE_REGISTER_SHAPE #data[:,1] == MSG_TYPE_REGISTER_SHAPE 

    regsysid = auxdata[idx] #data[idx][:,2] #should have more than
    #one value for more than one registration


    regshapeid = auxdata[shape_idx] #data[shape_idx][:,2]
    names = rawdata[idx] #data[idx][:,3].astype(np.uint8)
    
    dtype_data = rawdata[shape_idx]

    systems = dict()
    for sys in np.unique(regsysid):
        name = names[regsysid == sys].tostring()
        name = name[:-1] # Remove null terminator
        dtype = dtype_data[regshapeid == sys].tostring() #data[shape_idx][regshapeid == sys][:,3].astype(np.uint8).tostring()
        dtype = dtype[:-1] # Remove null terminator
        systems[sys] = name, dtype #name[:-1], dtype[:-1]
    #import pdb; pdb.set_trace()        
    return systems

def rowbyte(data, **kwargs):
    '''
    Parameters
    ----------
    data: np.array
        see docs for registrations for shape/dtype
    kwargs : dict
        see Docs for _split to see which kwargs are allowed

    Returns
    -------
    systems : dict
        In the dictionary, keys are the ID # of the system (assigned sequentially
        during registration time when the task is initializing). Values are
        np.array of shape (N, 2) where the columns are (timestamp, rawdata)
    '''
    if data.ndim < 2 or data.shape[1] != 4:
        data = _split(data, **kwargs)

    msgs = data[data[:,1] == MSG_TYPE_ROWBYTE]
    systems = dict()
    for i in np.unique(msgs[:,2]):
        systems[i] = msgs[msgs[:,2] == i][:,[0,-1]]
    return systems

def messages(data, **kwargs):
    '''
    Parse out any string messages sent byte-by-byte to the neural recording system

    Parameters
    ----------
    data : np.ndarray 
        Integer data and timestamps as stored in the neural data file when messages were sent during experiment
        OR, the result of the _split function

    Returns
    -------
    record array
        fields of 'time' and 'state' (message)
    '''
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

