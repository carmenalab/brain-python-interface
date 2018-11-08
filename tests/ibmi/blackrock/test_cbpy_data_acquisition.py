import sys
import time
from cerebus import cbpy

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

n_secs = 5
loop_time = 0.1
n_itrs = int(n_secs / loop_time)

for itr in range(n_itrs):
    t_start = time.time()
    print('\nitr %d of %d:' % (itr+1, n_itrs))

    # print 'calling cbpy.trial_event()'
    # result, trial = cbpy.trial_event(reset=True)
    # result, nsp_time = cbpy.time()
    print('calling cbpy.trial_continuous()')
    result, trial = cbpy.trial_continuous(reset=True)
    result, nsp_time = cbpy.time()
    for i in range(len(trial)):
        print('# samples:', len(trial[i][1]))

    print('result:', result)
    print('trial:', trial)
    print('')

    # print 'calling cbpy.time()'
    result, nsp_time = cbpy.time()
    print('result:', result)
    print('time:', nsp_time)
    print('')

    t_elapsed = time.time() - t_start
    if t_elapsed < loop_time:
        time.sleep(loop_time - t_elapsed)
    print('t_elapsed: %3.3f secs, loop time: %3.3f secs' % (t_elapsed, time.time()-t_start))

result = cbpy.close()
print('result:', result)
print('')
