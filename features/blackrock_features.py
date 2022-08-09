'''
Features for interacting with Blackrock's Neuroport neural recording system
'''

import time
import fnmatch
from .neural_sys_features import CorticalData, CorticalBMI
from riglib import blackrock

class RelayBlackrock(object):
    '''Sends full data directly into the Blackrock system.'''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' creates a sink for the NIDAQ card
        '''

        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        self.nidaq = sink_manager.start(self.ni_out)
        super(RelayBlackrock, self).init()

    @property
    def ni_out(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib.dio import nidaq
        print('nidaq.SendAll', nidaq.SendAll)
        return nidaq.SendAll

    @property    
    def blackrockfiles(self):
        '''Finds the blackrock files --- .nev and .nsx (.ns1 through .ns6) 
        --- that are most likely associated with the current task based on 
        the time at which the task ended and the "last modified" time of the 
        files located at /storage/blackrock/.
        '''
        import os, sys, glob, time
        if len(self.event_log) < 1:
            return None

        start = self.event_log[-1][2]
        
        files = []
        for ext in ['.nev', '.ns1', '.ns2', '.ns3', '.ns4', '.ns5', '.ns6']:
            pattern = "/storage/blackrock/*" + ext

            #matches = sorted(glob.glob(pattern), key=lambda f: abs(os.stat(f).st_mtime - start))
            matches = []
            for root, dirnames, filenames in os.walk('/storage/blackrock'):
                for filename in fnmatch.filter(filenames, '*' + ext):
                    matches.append(os.path.join(root, filename))

            sorted_matches = sorted(matches, key=lambda f: abs(os.stat(f).st_mtime - start))

            if len(sorted_matches) > 0:
                tdiff = os.stat(sorted_matches[0]).st_mtime - start
                if abs(tdiff) < 60:
                     files.append(sorted_matches[0])

        return files
    
    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method stops the NIDAQ sink after the FSM has finished running
        '''

        try:
            super(RelayBlackrock, self).run()
        finally:
            self.nidaq.stop()

            # Remotely stop the recording on the blackrock box
            import comedi
            import config
            import time
            com = comedi.comedi_open("/dev/comedi0")
            time.sleep(0.5)
            # strobe pin should already be low

            # set last data pin ("D15"; 16th pin) low
            comedi.comedi_dio_bitfield2(com, 0, 1, 0, 15)

            # set strobe pin high
            comedi.comedi_dio_bitfield2(com, 0, 1, 1, 16)

            # set strobe pin low
            comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)

    def set_state(self, condition, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.nidaq.sendMsg(condition)
        super(RelayBlackrock, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(RelayBlackrock, self).cleanup(database, saveid, **kwargs)
        time.sleep(2)
        for file_ in self.blackrockfiles:
            suffix = file_[-3:]  # e.g., 'nev', 'ns3', etc.
            # database.save_data(file_, "blackrock", saveid, True, False, custom_suffix=suffix)
            database.save_data(file_, "blackrock", saveid, False, False, suffix)

    @classmethod 
    def pre_init(cls, saveid=None, **kwargs):
        if saveid is not None:
            import comedi
            import config
            import time

            com = comedi.comedi_open("/dev/comedi0")
            # set strobe pin low
            comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)

            # set last data pin ("D15"; 16th pin) high
            comedi.comedi_dio_bitfield2(com, 0, 1, 1, 15)

            # set strobe pin high
            comedi.comedi_dio_bitfield2(com, 0, 1, 1, 16)

            # set strobe pin low
            comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)

            time.sleep(3)
            super(RelayPlexon, cls).pre_init(saveid=saveid, **kwargs)


class RelayBlackrockByte(RelayBlackrock):
    '''Relays a single byte (0-255) as a row checksum for when a data packet arrives.'''
    
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' checks that the experiment also uses the SaveHDF feature. 
        '''
    
        if not isinstance(self, SaveHDF):
            raise ValueError("RelayBlackrockByte feature only available with SaveHDF")
        
        super(RelayBlackrockByte, self).init()

    @property
    def ni_out(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import nidaq
        return nidaq.SendRowByte


class BlackrockData(object):
    '''Stream Blackrock neural data.'''

    blackrock_channels = None

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up DataSource objects for streaming from the Blackrock system.
        For LFP streaming, the data is stored as it is received.
        '''

        from riglib import blackrock, source

        if hasattr(self, "_neural_src_type") and hasattr(self, "_neural_src_kwargs") and hasattr(self, "_neural_src_system_type"):
            # for testing only!
            self.neurondata = self._neural_src_type(self._neural_src_system_type, **self._neural_src_kwargs)
        elif 'spike' in self.decoder.extractor_cls.feature_type:  # e.g., 'spike_counts'
            self.neurondata = source.DataSource(blackrock.Spikes, channels=self.blackrock_channels, send_data_to_sink_manager=False)
        elif 'lfp' in self.decoder.extractor_cls.feature_type:  # e.g., 'lfp_power'
            self.neurondata = source.MultiChanDataSource(blackrock.LFP, channels=self.blackrock_channels, send_data_to_sink_manager=True)
        else:
            raise Exception("Unknown extractor class, unable to create data source object!")

        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.neurondata)

        super(BlackrockData, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the 'neurondata' source before the FSM begins execution
        and stops it after the FSM has completed. 
        '''

        self.neurondata.start()
        try:
            super(BlackrockData, self).run()
        finally:
            self.neurondata.stop()


class BlackrockData(CorticalData):
    @property 
    def sys_module(self):
        return blackrock

class BlackrockBMI(CorticalBMI):
    @property 
    def sys_module(self):
        return blackrock    


# from riglib.bmi.bmi import Decoder
# class BlackrockBMI(BlackrockData, traits.HasTraits):
#     '''
#     Special case of BlackrockData which specifies a subset of channels to stream, i.e., the ones used by the Decoder
#     '''

#     decoder = traits.Instance(Decoder)

#     def init(self):
#         '''
#         Secondary init function. See riglib.experiment.Experiment.init()
#         Prior to starting the task, this 'init' decides which Blackrock channels to stream based on the channels in use by the decoder.
#         '''
#         print "HARDCODING BLACKROCK_CHANNELS -- CHANGE THIS!!"
#         self.blackrock_channels = range(1, 41) #self.decoder.units[:,0]
#         super(BlackrockBMI, self).init()
