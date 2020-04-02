import time
import numpy as np
from riglib.experiment import traits
import scipy.io as sio
from riglib.bmi import extractor


channels = np.arange(1, 97)
n_chan = len(channels)

extractor_cls = extractor.LFPMTMPowerExtractor

class BlackrockData(object):
# class BlackrockData(traits.HasTraits):
    '''Stream Blackrock neural data.'''

    def init(self):
        from riglib import blackrock, source

        if 'spike' in extractor_cls.feature_type:  # e.g., 'spike_counts'
            self.neurondata = source.DataSource(blackrock.Spikes, channels=channels)
        elif 'lfp' in extractor_cls.feature_type:  # e.g., 'lfp_power'
            self.neurondata = source.MultiChanDataSource(blackrock.LFP, channels=channels)
        else:
            raise Exception("Unknown extractor class, unable to create data source object!")

        try:
            super(BlackrockData, self).init()
        except:
            print("BlackrockData: running without a task")

    def run(self):
        self.neurondata.start()



if __name__ == '__main__':

    self = BlackrockData()
    self.init()
    self.run()

    n_secs = 60*10
    update_rate = 0.1
    N = int(n_secs / update_rate)

    samp_freq = 2000
    n_samp = int(n_secs * samp_freq)  # approx number of samples we'll collect per channel

    data = np.zeros((n_chan, 2*n_samp))
    idxs = np.zeros(n_chan)

    for k in range(N):
        t_start = time.time()

        new_data = self.neurondata.get_new(channels=channels)

        for row in range(n_chan):
            d = new_data[row]
            #print row, d.shape
            idx = idxs[row]
            data[row, idx:idx+len(d)] = d
            idxs[row] += len(d)

        t_elapsed = time.time() - t_start
        time.sleep(update_rate - t_elapsed)

    self.neurondata.stop()
     
    save_dict = dict()
    save_dict['data'] = data

    sio.savemat('cbpy_lfp_data.mat', save_dict)
