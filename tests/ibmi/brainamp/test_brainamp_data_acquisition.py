import time
import numpy as np
from riglib.experiment import traits
import scipy.io as sio
from riglib.bmi import extractor


#channels = [1, 2, 3, 4]
channels = ['AbdPolLo', 'ExtDig', 'ExtCU','Flex','PronTer','Biceps','Triceps','FrontDelt','MidDelt','BackDelt']
#channels = ['O1', 'O2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'T7', 'T8', 
#            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
#            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
#            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
#            '41', '42']
n_chan = len(channels)


class BrainAmpData(object):
    '''Stream BrainAmp neural data.'''

    def init(self):
        from riglib import source
        from riglib.brainamp import rda

        self.emgdata = source.MultiChanDataSource(rda.EMGData, channels=channels)


        try:
            super(BrainAmpData, self).init()
        except:
            print("BrainAmpData: running without a task")

    def run(self):
        self.emgdata.start()




if __name__ == '__main__':

    self = BrainAmpData()
    self.init()
    self.run()

    n_secs = 10
    update_rate = 1./10
    N = int(n_secs / update_rate)

    samp_freq = self.emgdata.source.update_freq
    n_samp = N * update_rate * samp_freq  # approx number of samples we'll collect per channel

    n_chan = len(channels)
    data = np.zeros((n_chan, 2*n_samp))
    idxs = np.zeros(n_chan)

    # print 'discarding initial data...',
    # t_start = time.time()
    # while time.time() - t_start < 5:
    #     # get new_data but don't do anything with it
    #     new_data = self.emgdata.get_new(channels=channels)
    #     time.sleep(0.005)
    # print 'done.'

    for k in range(N):
        t_start = time.time()

        new_data = self.emgdata.get_new(channels=channels)
        print(new_data[0].shape)
        print(new_data[0].dtype)

        for ts in new_data[0][:]['ts_arrival']:
            print(ts)


        # for row in range(n_chan):
        #     d = new_data[row]
        #     idx = idxs[row]
        #     data[row, idx:idx+len(d)] = d
        #     idxs[row] += len(d)

        t_elapsed = time.time() - t_start
        # print t_elapsed
        time.sleep(update_rate - t_elapsed)

    self.emgdata.stop()

    save_dict = dict()
    save_dict['data'] = data
    save_dict['channels'] = channels

    print('saving data...', end=' ')
    sio.matlab.savemat('emg_data.mat', save_dict)
    print('done.')

    print(data)
    print(idxs)
