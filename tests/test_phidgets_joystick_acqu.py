from Phidgets.Devices.InterfaceKit import InterfaceKit
import time

kit = InterfaceKit()

kit.openPhidget()
#kit.enableLogging(6,'phid_log.out')
kit.waitForAttach(2000)

s = dict()    
s['sens'] = np.zeros((1,2))
s['start_time'] = time.time()
sec_of_dat = 600
f_s = 60
err_ind = []
for i in range(sec_of_dat*f_s):
    s['tic'] = time.time()

    sensdat = np.zeros((1,2))
    try:
        sensdat[0,0] = kit.getSensorValue(0)/1000.
        sensdat[0,1] = kit.getSensorValue(1)/1000.
    except:
        print(time.time() - s['start_time'], i)
        print(kit.isAttached())
        err_ind.extend([i])

    try:
        print(kit.getSensorRawValue(2), kit.getSensorValue(2))
    except:
        print('novalue')

    s['sens'] = np.vstack((s['sens'], sensdat))
    left_over_time = np.max([0, 1000/60. - (time.time() - s['tic'])])
    time.sleep(left_over_time/1000.)
kit.closePhidget()
plt.plot(np.array(err_ind)/float(f_s))


