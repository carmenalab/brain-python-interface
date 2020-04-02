import time
from cerebus import cbpy

parameters = dict()
parameters['inst-addr']   = '192.168.137.128'
parameters['inst-port']   = 51001
parameters['client-addr'] = '192.168.137.255'
parameters['client-port'] = 51002

# setting kern.ipc.maxsockbuf to 8388608 on OS X seems fine, but specifying
#   that here causes error (-24) to occur
parameters['receive-buffer-size'] = 8388608

result, return_dict = cbpy.open(connection='default', parameter=parameters)
print('result:', result)
print('connection:', return_dict['connection'])
print('instrument:', return_dict['instrument'])
print('')


range_parameter = dict()
range_parameter['begin_channel'] = 0 #5
range_parameter['end_channel']   = 0 #8

buffer_parameter = dict()
buffer_parameter['absolute'] = True

# result, reset = cbpy.trial_config(range_parameter=range_parameter)  # works
# result, reset = cbpy.trial_config(buffer_parameter=buffer_parameter)  # doesn't work
result, reset = cbpy.trial_config(range_parameter=range_parameter, buffer_parameter=buffer_parameter)  # works
# result, reset = cbpy.trial_config()

print('result:', result)
print('reset:', reset)
print('')

n_secs = 10
loop_time = 0.1
n_itrs = int(n_secs / loop_time)

f = open('data.txt', 'w')

for itr in range(n_itrs):
    t_start = time.time()
    f.write('\nitr %d of %d:\n' % (itr+1, n_itrs))

    result, trial = cbpy.trial_event(reset=True)
    # print trial
    # # print timestamps only for one chan-unit
    # for list_ in trial:
    #     chan = list_[0]
    #     if chan == 5:
    #          for unit, unit_ts in enumerate(list_[1]['timestamps']):
    #             if unit == 0:
    #                 for timestamp in unit_ts:
    #                     if not got_first_timestamp:
    #                         first_timestap = timestamp
    #                         got_first_timestamp = True
    #                     f.write(str(timestamp/30000.) + '\n')
    for list_ in trial:
        chan = list_[0]
        for unit, unit_ts in enumerate(list_[1]['timestamps']):
            for timestamp in unit_ts:
                f.write(str((chan, unit, timestamp)) + '\n')
    
    f.write('\n' * 2)

    t_elapsed = time.time() - t_start
    if t_elapsed < loop_time:
        time.sleep(loop_time - t_elapsed)
    # print 't_elapsed: %3.3f secs, loop time: %3.3f secs' % (t_elapsed, time.time()-t_start)

result = cbpy.close()
f.close()
