import time
import numpy as np
import matplotlib.pyplot as plt
from riglib.experiment import traits
import scipy.io as sio

#remember to change "plexnet_softserver_oldfiles" back to "plexnet" in LFP.__init__
channels = [1, 2, 5, 9, 10, 33, 191, 192, 250, 256]
chan_offset = 512

# # remember to change "plexnet" to "plexnet_softserver_oldfiles" in LFP.__init__
# channels = [66, 67, 68, 91, 151, 191, 192]
# chan_offset = 0

class PlexonLFPData(traits.HasTraits):
    '''Stream neural LFP data from the Plexon system'''
    
    def init(self):
        from riglib import plexon, source
        print(channels)
        self.neurondata = source.MultiChanDataSource(plexon.LFP, channels=channels, chan_offset=chan_offset)
        try:
            super(PlexonLFPData, self).init()
        except:
            print("PlexonLFPData: running without a task")

    def run(self):
        self.neurondata.start()

if __name__ == '__main__':

    self = PlexonLFPData()
    self.init()
    self.run()

    n_secs = 15
    update_rate = 1./60
    N = int(n_secs / update_rate)

    samp_freq = 1000
    n_samp = N * update_rate * samp_freq  # approx number of samples we'll collect per channel

    n_chan = len(channels)
    data = np.zeros((n_chan, 2*n_samp))
    idxs = np.zeros(n_chan)

    print('discarding initial data...', end=' ')
    t_start = time.time()
    while time.time() - t_start < 5:
        # get new_data but don't do anything with it
        new_data = self.neurondata.get_new(channels=channels)
        time.sleep(0.005)
    print('done.')

    for k in range(N):
        t_start = time.time()

        new_data = self.neurondata.get_new(channels=channels)
        print(new_data[0].shape)

        for row in range(n_chan):
            d = new_data[row]
            idx = idxs[row]
            data[row, idx:idx+len(d)] = d
            idxs[row] += len(d)

        t_elapsed = time.time() - t_start
        # print t_elapsed
        time.sleep(update_rate - t_elapsed)

    self.neurondata.stop()

    save_dict = dict()
    save_dict['data'] = data
    save_dict['channels'] = channels
    # save_dict['n_garbage'] = n_garbage

    print('saving data...', end=' ')
    sio.matlab.savemat('lfp_data_0222_8pm_1.mat', save_dict)
    print('done.')

    print(data)
    print(idxs)
