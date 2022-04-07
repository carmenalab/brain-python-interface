'''
Generic features for interfacing with cortical recording systems (Plexon Omniplex, Blackrock Neuroport, etc.)
'''
from riglib import source, sink
from riglib.experiment import traits
from riglib.bmi.bmi import Decoder 
import time

class CorticalData(object):
    '''
    Feature for streaming data from Omniplex/Neuroport recording systems
    '''
    cortical_channels = None
    register_with_sink_manager = False
    send_data_to_sink_manager = False

    def init(self):
        sys_module = self.sys_module # e.g., riglib.plexon, riglib.blackrock

        kwargs = dict(send_data_to_sink_manager=self.send_data_to_sink_manager, channels=self.cortical_channels)
        if hasattr(self, "neural_src_kwargs"):
            kwargs.update(self.neural_src_kwargs)

        if hasattr(self, "_neural_src_type") and hasattr(self, "_neural_src_kwargs") and hasattr(self, "_neural_src_system_type"):
            # for testing only!
            self.neurondata = self._neural_src_type(self._neural_src_system_type, **self._neural_src_kwargs)
        elif 'spike' in self.decoder.extractor_cls.feature_type:  # e.g., 'spike_counts'
            self.neurondata = source.DataSource(sys_module.Spikes, **kwargs)
        elif 'lfp' in self.decoder.extractor_cls.feature_type:  # e.g., 'lfp_power'
            self.neurondata = source.MultiChanDataSource(sys_module.LFP, **kwargs)
        else:
            raise Exception("Unknown extractor class, unable to create data source object!")

        if self.register_with_sink_manager:
            sink_manager = sink.SinkManager.get_instance()
            sink_manager.register(self.neurondata)            

        super(CorticalData, self).init()

    @property 
    def sys_module(self):
        raise NotImplementedError("You must create a child class which specifies the recording system!")

    def run(self):
        self.neurondata.start()
        time.sleep(1) # give the datasource time to start
        try:
            super(CorticalData, self).run()
        finally:
            self.neurondata.stop()


class CorticalBMI(CorticalData, traits.HasTraits):
    '''
    Special case of CorticalData which specifies a subset of channels to stream, i.e., the ones used by the Decoder
    May not be available for all recording systems. 
    '''
    decoder = traits.InstanceFromDB(Decoder, bmi3d_db_model='Decoder', bmi3d_query_kwargs=dict())

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets the channels to be the channels of the decoder
        so that the PlexonData source only grabs the channels actually used by the decoder. 
        '''
        self.cortical_channels = [int(ch) for ch in self.decoder.units[:,0]]
        super(CorticalBMI, self).init()
