import time
import numpy as np
from riglib.experiment import traits
import scipy.io as sio
from riglib.bmi import extractor


channels = list(range(33))
n_chan = len(channels)

extractor_cls = extractor.BinnedSpikeCountsExtractor
# extractor_cls = extractor.LFPMTMPowerExtractor

class BlackrockData(object):
    '''Stream Blackrock neural data.'''

    def init(self, data_type, channels):
        from riglib import blackrock, source

        if data_type is 'spike':  # e.g., 'spike_counts'
            self.neurondata = source.DataSource(blackrock.Spikes, channels=channels)
        elif data_type is 'lfp':  # e.g., 'lfp_power'
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

    # f = open('data.txt', 'w')

    self = BlackrockData()
    self.init('spike',channels)
    self.run()

    n_secs = 30
    update_rate = 1./60
    N = int(n_secs / update_rate)

    idxs = dict()
    data = dict()
    for chan in channels:
        idxs[chan] = 0
        data[chan] = np.zeros((2, 400))

    for k in range(N):
        t_start = time.time()
        # f.write('Iteration: %d\n' % k)

        new_data = self.neurondata.get()
        for (ts, chan, unit) in zip(new_data['ts'], new_data['chan'], new_data['unit']):
            #if chan in channels:
            data[chan][0, idxs[chan]] = ts * 30000
            data[chan][1, idxs[chan]] = unit
            idxs[chan] += 1
                #print (ts, chan, unit)
            #else:
                #pass
                # print 'received data on unwanted channel:', chan

        # print new_data

        # f.write(str(new_data))
        # f.write('\n\n')

        t_elapsed = time.time() - t_start
        time.sleep(update_rate - t_elapsed)

    self.neurondata.stop()
     
    save_dict = dict()
    for chan in channels:
        save_dict['chan' + str(chan)] = data[chan]

    #sio.savemat('cbpy_spike_data.mat', save_dict)

    # print save_dict

    # f.close()

    for i in range(32):
        if i == 0:
            assert np.sum(data[i][0, :]) == 0.
            assert np.sum(data[i][1, :]) == 0.
        elif i == 2: 
            assert len(np.array([j for j in data[i][1, :] if j in [1, 2, 3]]))==0
        elif i == 12:
            assert len(np.array([j for j in data[i][1, :] if j in [2]]))==0
        elif i == 24: 
            assert len(np.array([j for j in data[i][1, :] if j in [1, 2, 3]]))==0
        else:
            assert len(np.array([j for j in data[i][1, :] if j not in [1, 2, 3, 4, 10, 0]]))==0