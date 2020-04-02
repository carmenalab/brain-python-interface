import sys
import time
import numpy as np
from cerebus import cbpy
import pprint
import tabulate

# on my setup, nPlay streams back data on channel x as coming from channel x+4
# (e.g., spike data on channel 1 saved in the .nev file is streamed by nPlay
#  as if it's coming from channel 5)
nPlay_chan_offset = 0

parameters = dict()
parameters['inst-addr']   = '192.168.137.128'
parameters['inst-port']   = 51001
parameters['client-port'] = 51002

if sys.platform == 'darwin':  # OS X
    print('Using OS X settings for cbpy')
    parameters['client-addr'] = '255.255.255.255'
else:  # linux
    print('Using linux settings for cbpy')
    parameters['client-addr'] = '192.168.137.255'
    parameters['receive-buffer-size'] = 8388608

result, return_dict = cbpy.open(connection='default', parameter=parameters)
print('result:', result)
print('connection:', return_dict['connection'])
print('instrument:', return_dict['instrument'])
print('')


buffer_parameter = {'absolute': True}
result, reset = cbpy.trial_config(buffer_parameter=buffer_parameter)
print('result:', result)
print('reset:', reset)
print('')

n_secs = 20
loop_time = 0.1
n_itrs = int(n_secs / loop_time)

last_nsp_time = 0

for itr in range(n_itrs):
    t_start = time.time()
    print('-' * 79)
    print('\nitr %d of %d:' % (itr+1, n_itrs))
    print('')

    # print 'calling cbpy.trial_event(), followed by cbpy.time()'
    # print ''
    trial_event_result, trial = cbpy.trial_event(reset=True)
    time_result, nsp_time = cbpy.time()

    n_ts = 0
    for list_ in trial:
        for unit, unit_ts in enumerate(list_[1]['timestamps']):
            n_ts += len(unit_ts)

    data = np.zeros((n_ts, 3), dtype=np.int32)  # 3 columns: timestamp, chan, unit
    idx = 0
    for list_ in trial:
        chan = list_[0] - nPlay_chan_offset
        for unit, unit_ts in enumerate(np.array(list_[1]['timestamps'])):
            if len(unit_ts) > 0:
                for ts in unit_ts:
                    if ts < last_nsp_time:
                        args = (ts, chan, unit, last_nsp_time)
                        print('Warning: timestamp %d for (chan,unit)=(%d,%d) is less than last NSP time of %d' % args)

                inds = list(range(idx, idx + len(unit_ts)))

                data[inds, 0] = unit_ts
                data[inds, 1] = chan
                data[inds, 2] = unit

                idx += len(unit_ts)
        

    # print 'trial_event result:', trial_event_result
    print('')
    print('trial data:\n')

    if data.shape[0] > 0:
        data = data[np.argsort(data[:,0]), :]
        
        headers = ['timestamp', 'channel', 'unit']
        print(tabulate.tabulate(data, headers=headers))
        print('')

    # print 'time result:', time_result
    print('NSP time:', int(nsp_time))
    print('')

    last_nsp_time = nsp_time

    t_elapsed = time.time() - t_start
    if t_elapsed < loop_time:
        time.sleep(loop_time - t_elapsed)
    # print 't_elapsed: %3.3f secs, loop time: %3.3f secs' % (t_elapsed, time.time()-t_start)

result = cbpy.close()
# print 'result:', result
# print ''
