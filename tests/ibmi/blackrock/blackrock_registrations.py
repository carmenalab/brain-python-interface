import os
import numpy as np
from riglib.dio import parse
import subprocess
import matplotlib.pyplot as plt

def sys_eq(sys1, sys2):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    return sys1 in [sys2, sys2[1:]]

def load_file(nev_fname,):

    # nev_fname = '/storage/blackrock/20140707-171012/20140707-171012-002.nev'
    # nev_fname = 'test20151222_65_te1276.nev'

    nev_hdf_fname = nev_fname + '.hdf'
    
    if not os.path.isfile(nev_hdf_fname):
        # convert .nev file to hdf file using Blackrock's n2h5 utility
        subprocess.call(['n2h5', nev_fname, nev_hdf_fname])

    import tables
    nev_hdf = tables.openFile(nev_hdf_fname)
    hdf_fname = nev_fname[:-4]+'.hdf'
    hdf = tables.openFile(hdf_fname)

    return nev_hdf, hdf

def parse_nev_hdf(nev_hdf):

    ts = nev_hdf.root.channel.digital00001.digital_set[:]['TimeStamp']
    msgs = nev_hdf.root.channel.digital00001.digital_set[:]['Value']

    # copied from riglib/dio/parse.py
    msgtype_mask = parse.msgtype_mask 
    auxdata_mask = parse.auxdata_mask 
    rawdata_mask = parse.rawdata_mask

    msgtype = np.right_shift(np.bitwise_and(msgs, msgtype_mask), 8).astype(np.uint8)
    # auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 11).astype(np.uint8)
    rawdata = np.bitwise_and(msgs, rawdata_mask)

    # data is an N x 4 matrix that will be the argument to parse.registrations()
    data = np.vstack([ts, msgtype, auxdata, rawdata]).T

    # get system registrations
    reg = parse.registrations(data)

    return data, reg

def compare_hdfs(data, reg, hdf):
    system_list = [system[0] for k,system in list(reg.items())]

    f, ax = plt.subplots(nrows = len(system_list))
    if len(system_list)==1:
        ax = [ax]

    for i_s, sys_name in enumerate(system_list):
        syskey = None
        for key, system in list(reg.items()):
            
            #match each system in the nev_hdf to a table in the normal hdf:
            if sys_eq(system[0], sys_name):
                syskey = key
                break

        if syskey is None:
            raise Exception('No source registration saved in the file!')

        rows = parse.rowbyte(data)[syskey][:,0]
        timestamps = rows / 30000.
        
        print(sys_name, 'rows in nev_hdf: ', len(timestamps), 'rows in hdf: ', len(hdf.get_node('/'+sys_name)))

        tab = hdf.get_node('/'+sys_name)
        if sys_name == 'brainamp':
            ts = np.squeeze(tab[:]['chan1']['ts_arrival'])
        else:
            ts = np.squeeze(tab[:]['ts'])


        ax[i_s].plot(np.diff(timestamps), label='.nev')
        ax[i_s].plot(np.diff(ts), label='.hdf')
        #ax[i_s].plot(timestamps-timestamps[0], label='.nev')
        #ax[i_s].plot(ts-ts[0], label='.hdf')

        ax[i_s].set_title(sys_name)
        ax[i_s].legend()
        
    plt.tight_layout()



