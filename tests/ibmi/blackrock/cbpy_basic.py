'''Test script to receive and print spike/event data using cbpy'''

from CereLink import cbpy
import time

parameters = dict()
parameters['inst-addr']   = '192.168.137.128'
parameters['inst-port']   = 51001
parameters['client-addr'] = '255.255.255.255'
parameters['client-port'] = 51002

print('calling cbpy.open()')
return_dict = cbpy.open('default', parameters)
print('connection:', return_dict['connection'])
print('instrument:', return_dict['instrument'])

try:
    print('calling cbpy.trial_config()')
    return_dict = cbpy.trial_config(True)
    # trial_config doesn't return a dict as advertised -- instead, returns a boolean (?)
    # print 'label:', return_dict['label']
    # print 'enabled:', return_dict['enabled']
    # print 'valid_unit:', return_dict['valid_unit']

    n_itrs = 20
    loop_time = 0.1

    for itr in range(n_itrs):
        t_start = time.time()
        print('\nitr %d of %d:' % (itr+1, n_itrs))

        print('calling cbpy.trial_event()')
        # returns a list of tuples: (channel, digital_events) or (channel, unit0_ts, ..., unitN_ts)
        data = cbpy.trial_event(True)
        print('data:', data)

        t_elapsed = time.time() - t_start
        if t_elapsed < loop_time:
            time.sleep(loop_time - t_elapsed)
        print('t_elapsed: %3.3f secs, loop time: %3.3f secs' % (t_elapsed, time.time()-t_start))

finally:
    print('calling cbpy.close()')
    cbpy.close()
