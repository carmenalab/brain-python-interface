import numpy as np
import pickle
import socket
import struct

ts_dtype = [('ts', float), ('chan', np.int32), ('unit', np.int32)]
class DataSourceSimSpikes():
    def __init__(self):
        pass
    def get(self):
        ip_addr = '127.0.0.1'
        MAX_ELECTRODES = 128
        MAX_SPIKES_PER_ELECTRODE = 4
        spike_request_port = 22301

        addr = (ip_addr, spike_request_port)
        soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        soc.sendto('spike_request', addr)
        for k in range(10):
            try: 
                # read the number of spike time stamps
                data, addr = soc.recvfrom(8)
                n = int(struct.unpack('d', data)[0])
                #print n
                
                data, addr = soc.recvfrom(16*n)
                #print len(data)
                if len(data) == 16*n:
                    X = struct.unpack('f'*(n*4), data)
                    #print X
                    break
            except:
                print "Re-calling the same spikes"

        soc.close()

        # re-inflate the array
        ts = np.array(X).reshape(-1, 4)

        # convert to "python" format
        n_ts = ts.shape[0]
        #spike_inds = filter(lambda k: ts[k,0] == 1, range(n_ts))
        ts_ls = [(ts[k,3], ts[k,1], ts[k,2]) for k in range(n_ts)]

        return np.array(ts_ls, dtype=ts_dtype)

if __name__ == '__main__':
    source = DataSourceSimSpikes()
    ts = source.get()
