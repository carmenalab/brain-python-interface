import time
import numpy as np
from riglib.experiment import traits
import scipy.io as sio
from riglib.bmi import extractor


channels = [5, 6, 7, 8, 9, 10, 11, 12]
n_chan = len(channels)

extractor_cls = extractor.BinnedSpikeCountsExtractor
# extractor_cls = extractor.LFPMTMPowerExtractor

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
            print "BlackrockData: running without a task"

    def run(self):
        self.neurondata.start()



if __name__ == '__main__':

    self = BlackrockData()
    self.init()
    self.run()

    n_secs = 15
    update_rate = 1./60
    N = int(n_secs / update_rate)

    data = dict()
    for chan in channels:
        data[chan] = np.zeros((2, 400))
    idxs = np.zeros(n_chan)

    for k in range(N):
        t_start = time.time()

        data = self.neurondata.get()
        for (ts, chan, unit) in zip(new_data['ts'], new_data['chan'], new_data['unit']):
            data[chan][0, idxs[chan]] = ts
            data[chan][1, idxs[chan]] = unit
            idxs[chan] += 1

        # print new_data

        t_elapsed = time.time() - t_start
        time.sleep(update_rate - t_elapsed)

    self.neurondata.stop()
     
    save_dict = dict()
    for chan in channels:
        save_dict[str(chan)] = data[chan]

    sio.matlab.savemat('cbpy_spike_data.mat', save_dict)
