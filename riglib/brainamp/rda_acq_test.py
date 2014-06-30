import time
import numpy as np
from riglib.experiment import traits
import scipy.io as sio
from riglib.bmi import extractor


channels = [1, 2, 3, 4]
n_chan = len(channels)


class BrainAmpData(object):
# class BrainAmpData(traits.HasTraits):
    '''Stream BrainAmp neural data.'''

    def init(self):
        from riglib import brainamp, source

        self.emgdata = source.MultiChanDataSource(brainamp.EMG, channels=channels)

        try:
            super(BrainAmpData, self).init()
        except:
            print "BrainAmpData: running without a task"

    def run(self):
        self.emgdata.start()



if __name__ == '__main__':

    # f = open('data.txt', 'w')

    self = BrainAmpData()
    self.init()
    self.run()

    n_secs = 15
    update_rate = 1./10
    N = int(n_secs / update_rate)

    samp_freq = 1000
    n_samp = N * update_rate * samp_freq  # approx number of samples we'll collect per channel

    n_chan = len(channels)
    data = np.zeros((n_chan, 2*n_samp))
    idxs = np.zeros(n_chan)

    print 'discarding initial data...',
    t_start = time.time()
    while time.time() - t_start < 5:
        # get new_data but don't do anything with it
        new_data = self.emgdata.get_new(channels=channels)
        time.sleep(0.005)
    print 'done.'

    for k in range(N):
        t_start = time.time()

        new_data = self.emgdata.get_new(channels=channels)
        print new_data[0].shape

        for row in range(n_chan):
            d = new_data[row]
            idx = idxs[row]
            data[row, idx:idx+len(d)] = d
            idxs[row] += len(d)

        t_elapsed = time.time() - t_start
        # print t_elapsed
        time.sleep(update_rate - t_elapsed)

    self.emgdata.stop()

    save_dict = dict()
    save_dict['data'] = data
    save_dict['channels'] = channels
    # save_dict['n_garbage'] = n_garbage

    print 'saving data...',
    sio.matlab.savemat('emg_data.mat', save_dict)
    print 'done.'

    print data
    print idxs
