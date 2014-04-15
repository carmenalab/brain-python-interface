import time
import numpy as np
from riglib.experiment import traits
import scipy.io as sio
from riglib.bmi import extractor


channels = [5, 6, 7, 8]

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

    f = open('data.txt', 'w')

    self = BlackrockData()
    self.init()
    self.run()

    n_secs = 15
    update_rate = 1./60
    N = int(n_secs / update_rate)

    for k in range(N):
        t_start = time.time()
        f.write('Iteration: %d\n' % k)

        new_data = self.neurondata.get()
        f.write(str(new_data))

        t_elapsed = time.time() - t_start

        f.write('\n\n')

        if t_elapsed < update_rate:
            time.sleep(update_rate - t_elapsed)

    self.neurondata.stop()

    f.close()
