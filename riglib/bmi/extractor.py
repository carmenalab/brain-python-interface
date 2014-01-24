import numpy as np

# object that gets the data that it needs (e.g., spikes, LFP, etc.) from the neural data source and 
# extracts features from it
class FeatureExtractor(object):
    '''Docstring.'''

    def __init__(self, task):
        raise NotImplementedError  # subclasses need to implement this

    def get_neural_data(self):
        raise NotImplementedError  # subclasses need to implement this

    def extract_features(self):
        raise NotImplementedError  # subclasses need to implement this



class BinnedSpikeCountsExtractor(FeatureExtractor):
    '''Docstring.'''

    # method takes the task as input and stores whatever aspects of the task
    # it will need for the future
    def __init__(self, task):
        self.task = task  # eventually get rid of this so that we don't need to store self.task
        self.n_subbins = task.n_subbins
        self.units = task.decoder.units

    def get_neural_data(self):
        # eventually change this so that this extractor gets ts directly from 
        # the neural data source (e.g., SpikeBMI feature), and not from the task
        self.ts = self.task.get_spike_ts()

    def extract_features(self):
        ts = self.ts
        
        if len(ts) == 0:
            counts = np.zeros([len(self.units), self.n_subbins])
            bin_edges = np.array([np.nan, np.nan])
        else:
            min_ind = np.argmin(ts['ts'])
            max_ind = np.argmax(ts['ts'])
            bin_edges = np.array([ts[min_ind]['ts'], ts[max_ind]['ts']])
            if self.n_subbins > 1:
                subbin_edges = np.linspace(self.task.last_get_spike_counts_time, start_time, self.n_subbins+1)
                subbin_edges[0] -= 1 # Include any latent spikes that somehow got lost..
                subbin_inds = np.digitize(ts['arrival_ts'], subbin_edges)
                counts = np.vstack([bin_spikes(ts[subbin_inds == k], self.units) for k in range(1, self.n_subbins+1)]).T
            else:
                counts = bin_spikes(ts, self.units).reshape(-1, 1)

        # how to avoid setting this task_data variable from here?
        # don't want to simply return bin_edges because we don't expect every
        # feature extractor's extract_feature method to return that variable
        # (e.g., all LFP feature extractors)
        self.task.task_data['bin_edges'] = bin_edges

        # if len(ts) > 0 and task.verbose_plexnet:
        #     print "Largest ts observed: %f", (task.largest_ts - task.task_start_time)
        #     late_spikes = ts[ts['arrival_ts'] < task.last_get_spike_counts_time]
        #     if len(late_spikes) > 0:
        #         print "spikes that were before the first boundary:"
        #         print late_spikes['arrival_ts'] - task.task_start_time
        #         packet_error = np.all(late_spikes['arrival_ts'] == task.largest_ts)
        #         print packet_error
        #         if not packet_error:
        #             print task.last_get_spike_counts_time - task.task_start_time
        #         print "threading error of ", (task.last_get_spike_counts_time - late_spikes[0]['arrival_ts'])
        #     task.largest_ts = max(ts['arrival_ts'])
        #     print

        neural_features = counts
        return neural_features



# this function used to be part of the Decoder class
# but has now been moved here
def bin_spikes(ts, units, max_units_per_channel=13):
    '''
    Count up the number of BMI spikes in a list of spike timestamps.
    '''
    unit_inds = units[:,0]*max_units_per_channel + units[:,1]
    edges = np.sort(np.hstack([unit_inds - 0.5, unit_inds + 0.5]))
    spiking_unit_inds = ts['chan']*max_units_per_channel + ts['unit']
    counts, _ = np.histogram(spiking_unit_inds, edges)
    return counts[::2]

        



class LFPPowerExtractor(object):
    pass




