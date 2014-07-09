import time
from riglib.bmi import state_space_models

ismore_ss = state_space_models.StateSpaceIsMore()
# channels = ismore_ss.state_names
channels = ['aa_px', 'aa_py']

class IsMoreFeedbackData(object):

    def init(self):
        from riglib import blackrock, source

        self.feedback_source = source.MultiChanDataSource(blackrock.FeedbackData, channels=channels)

        try:
            super(IsMoreFeedbackData, self).init()
        except:
            print 'IsMoreFeedbackData: running without a task'

    def run(self):
        self.feedback_source.start()


if __name__ == '__main__':

    self = IsMoreFeedbackData()
    self.init()
    self.run()

    n_secs = 15
    update_rate = 0.2
    N = int(n_secs / update_rate)

    # samp_freq = 1000
    # n_samp = int(n_secs * samp_freq)  # approx number of samples we'll collect per channel

    # data = np.zeros((n_chan, 2*n_samp))
    # idxs = np.zeros(n_chan)

    for k in range(N):
        t_start = time.time()

        new_data = self.feedback_source.get_new(channels=channels)
        print new_data
        print ''

        # for row in range(n_chan):
        #     d = new_data[row]
        #     idx = idxs[row]
        #     data[row, idx:idx+len(d)] = d
        #     idxs[row] += len(d)

        t_elapsed = time.time() - t_start
        time.sleep(update_rate - t_elapsed)

    self.feedback_source.stop()
     
    # save_dict = dict()
    # save_dict['data'] = data

    # sio.matlab.savemat('cbpy_lfp_data.mat', save_dict)