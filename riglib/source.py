'''
Generic data source module. Sources run in separate processes and continuously collect/save
data and interact with the main process through the methods here.
'''
import time
import inspect
import traceback
import multiprocessing as mp
from multiprocessing import sharedctypes as shm

import numpy as np
from .mp_proxy import FuncProxy, RPCProcess

from . import sink

class DataSourceSystem(object):
    '''
    Abstract base class for use with the generic DataSource infrastructure. Requirements:
        1) the class must have an attribute named 'dtype' which represents 
           the data type of the source data. The datatype *cannot* change!
        2) the class must have an attribute named 'update_freq' which specifies 
           the frequency at which new data samples will be ready for retrieval.
        3) 'start' method--no arguments
        4) 'stop' method--no arguments
        5) 'get' method--should return a single output argument
    '''
    dtype = np.dtype([])
    update_freq = 1
    def start(self):
        '''
        Initialization for the source
        '''
        pass

    def stop(self):
        '''
        Code to run when the data source is to be stopped
        '''
        pass

    def get(self):
        '''
        Retrieve the current data available from the source. 
        '''
        pass


class DataSource(RPCProcess):
    def __init__(self, source, bufferlen=10, name=None, send_data_to_sink_manager=True, source_kwargs=dict(), **kwargs):
        '''
        Parameters
        ----------
        source: class compatible with DataSourceSystem
            Class to be instantiated as the "system" with changing data values. 
        bufferlen: float
            Number of seconds long to make the ringbuffer. Seconds are converted to number 
            of samples based on the 'update_freq' attribute of the source
        name: string, optional, default=None
            Name of the sink, i.e., HDF table. If one is not provided, it will be inferred based
            on the name of the source module
        send_data_to_sink_manager: boolean, optional, default=True
            Flag to indicate whether data should be saved to a sink (e.g., HDF file)
        kwargs: optional keyword arguments
            Passed to the source during object construction if any are specified

        Returns
        -------
        DataSource instance
        '''
        super(DataSource, self).__init__(**kwargs)
        if name is not None:
            self.name = name
        else:
            self.name = source.__module__.split('.')[-1]
        self.filter = None
        self.target_class = self.source = source
        self.target_kwargs = self.source_kwargs = source_kwargs #kwargs
        self.bufferlen = bufferlen
        self.max_len = bufferlen * int(self.source.update_freq)
        self.slice_size = self.source.dtype.itemsize
        
        self.lock = mp.Lock()
        self.idx = shm.RawValue('l', 0)
        self.data = shm.RawArray('c', self.max_len * self.slice_size)
        # self.pipe, self._pipe = mp.Pipe()
        # self.cmd_event = mp.Event()
        # self.status = mp.Value('b', 1)
        self.stream = mp.Event()
        self.last_idx = 0
        self.streaming = False


        # in DataSource.run, there is a call to "self.sinks.send(...)",
        # but if the DataSource was never registered with the sink manager,
        # then this line results in unnecessary IPC
        # so, set send_data_to_sink_manager to False if you want to avoid this
        self.send_data_to_sink_manager = send_data_to_sink_manager

    def target_constr(self):
        try:
            self.target = self.target_class(**self.target_kwargs)
            self.target.start()
        except Exception as e:
            print("source.DataSource.run: unable to start source!")
            print(e)

            import io
            err = io.StringIO()
            self.log_error(err, mode='a')
            err.seek(0)

            self.status.value = -1

        self.streaming = True

    def loop_task(self):
            size = self.slice_size

            if self.stream.is_set():
                self.stream.clear()
                self.streaming = not self.streaming
                if self.streaming:
                    self.idx.value = 0
                    self.target.start()
                else:
                    self.target.stop()

            if self.streaming:
                data = self.target.get()
                if self.send_data_to_sink_manager:
                    sink_manager = sink.SinkManager.get_instance()
                    sink_manager.send(self.name, data)
                if data is not None:
                    try:
                        self.lock.acquire()
                        i = self.idx.value % self.max_len
                        self.data[i*size:(i+1)*size] = np.array(data).tobytes()
                        self.idx.value += 1
                        self.lock.release()
                    except Exception as e:
                        print("source.DataSource.run, exception saving data to ring buffer")
                        print(e)
            else:
                time.sleep(.001)        

    def target_destr(self, ret_status, msg):
        # stop the system once self.status.value has been set to a negative number
        self.target.stop()

    def get(self, all=False, **kwargs):
        '''
        Retreive data from the remote process

        Parameters
        ----------
        all : boolean, optional, default=False
            If true, returns all the data currently available. Since a finite buffer is used, 
            this is NOT the same as all the data observed. (see 'bufferlen' in __init__ for buffer size)
        kwargs : optional kwargs 
            To be passed to self.filter, if it is listed

        Returns
        -------
        np.recarray 
            Datatype of record array is the dtype of the DataSourceSystem
        '''
        if self.status.value <= 0:
            raise Exception('\n\nError starting datasource: %s\n\n' % self.name)
            
        self.lock.acquire()
        i = (self.idx.value % self.max_len) * self.slice_size
        if all:
            if self.idx.value < self.max_len:
                data = self.data[:i]
            else:
                data = self.data[i:]+self.data[:i]
        else:
            mlen = min((self.idx.value - self.last_idx), self.max_len)
            last = ((self.idx.value - mlen) % self.max_len) * self.slice_size
            if last > i:
                data = self.data[last:] + self.data[:i]
            else:
                data = self.data[last:i]
            
        self.last_idx = self.idx.value
        self.lock.release()
        try:
            data = np.frombuffer(data, dtype=self.source.dtype)
        except:
            print("can't get fromstring...")

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    def pause(self):
        '''
        Used to toggle the 'streaming' variable in the remote "run" process 
        '''
        self.stream.set()


class MultiChanDataSource(mp.Process):
    '''
    Multi-channel version of 'DataSource'
    '''
    def __init__(self, source, bufferlen=5, name=None, send_data_to_sink_manager=False, **kwargs):
        '''
        Parameters
        ----------
        source: class 
            lower-level class for interacting directly with the incoming data (e.g., plexnet)
        bufferlen: int
            Constrains the maximum amount of data history stored by the source
        name: string, optional, default=None
            Name of the sink, i.e., HDF table. If one is not provided, it will be inferred based
            on the name of the source module
        send_data_to_sink_manager: boolean, optional, default=False
            Flag to indicate whether data should be saved to a sink (e.g., HDF file)            
        kwargs: dict, optional, default = {}
            For the multi-channel data source, you MUST specify a 'channels' keyword argument
            Note that kwargs['channels'] does not need to a list of integers,
            it can also be a list of strings.
        '''

        super(MultiChanDataSource, self).__init__()
        if name is not None:
            self.name = name
        else:
            self.name = source.__module__.split('.')[-1]
        self.filter = None
        self.source = source
        self.source_kwargs = kwargs
        self.bufferlen = bufferlen
        self.max_len = int(bufferlen * self.source.update_freq)
        self.channels = kwargs['channels']
        self.chan_to_row = dict()
        for row, chan in enumerate(self.channels):
            self.chan_to_row[chan] = row
        
        self.n_chan = len(self.channels)
        dtype = self.source.dtype  # e.g., np.dtype('float') for LFP
        self.slice_size = dtype.itemsize
        self.idxs = shm.RawArray('l', self.n_chan)
        self.last_read_idxs = np.zeros(self.n_chan, dtype='int')
        rawarray = shm.RawArray('c', self.n_chan * self.max_len * self.slice_size)


        self.data = np.frombuffer(rawarray, dtype).reshape((self.n_chan, self.max_len))

        

        #self.fo2 = open('/storage/rawdata/test_rda_get.txt','w')
        #self.fo3 = open('/storage/rawdata/test_rda_run.txt','w')


        self.lock = mp.Lock()
        self.pipe, self._pipe = mp.Pipe()
        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1)
        self.stream = mp.Event()
        self.data_has_arrived = mp.Value('b', 0)

        self.methods = set(n for n in dir(source) if inspect.ismethod(getattr(source, n)))

        self.send_data_to_sink_manager = send_data_to_sink_manager
        if self.send_data_to_sink_manager:
            self.send_to_sinks_dtype = np.dtype([('chan'+str(chan), dtype) for chan in kwargs['channels']])
            self.next_send_idx = mp.Value('l', 0)
            self.wrap_flags = shm.RawArray('b', self.n_chan)  # zeros/Falses by default
            self.supp_hdf_file = kwargs['supp_file']



    def register_supp_hdf(self):
        try:
            from ismore.brainamp import brainamp_hdf_writer
        except:
            from riglib.ismore import brainamp_hdf_writer
        self.supp_hdf = brainamp_hdf_writer.BrainampData(self.supp_hdf_file, self.channels, self.send_to_sinks_dtype)


    def verify_data_arrival(self):
        try:
            from ismore.brainamp.brainamp_features import verify_data_arrival
        except:
            from riglib.ismore.brainamp.brainamp_features import verify_data_arrival

        


    def start(self, *args, **kwargs):
        '''
        From Python's docs on the multiprocessing module:
            Start the process's activity.
            This must be called at most once per process object. It arranges for the object's run() method to be invoked in a separate process.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.sinks = sink_manager = sink.SinkManager.get_instance()
        super(MultiChanDataSource, self).start(*args, **kwargs)

    def run(self):
        '''
        Main function executed by the mp.Process object. This function runs in the *remote* process, not in the main process
        '''
        print(("Starting datasource %r" % self.source))
        if self.send_data_to_sink_manager:
            print(("Registering Supplementary HDF file for datasource %r" % self.source))
            self.register_supp_hdf()

        try:
            system = self.source(**self.source_kwargs)
            system.start()

        except Exception as e:
            print(e)
            self.status.value = -1

        streaming = True
        size = self.slice_size
        while self.status.value > 0:
            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                self.lock.acquire()
                try:
                    if cmd == "getattr":
                        ret = getattr(system, args[0])
                    else:
                        ret = getattr(system, cmd)(*args, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    ret = e
                self.lock.release()
                self._pipe.send(ret)
                self.cmd_event.clear()

            if self.stream.is_set():
                self.stream.clear()
                streaming = not streaming
                if streaming:
                    self.idx.value = 0
                    system.start()
                else:
                    system.stop()

            if streaming:
                # system.get() must return a tuple (chan, data), where: 
                #   chan is the the channel number
                #   data is a numpy array with a dtype (or subdtype) of
                #   self.source.dtype
                #print 'before get'
                

                chan, data = system.get()
                #self.fo3.write(str(data[0][0]) + ' ' + str(time.time()) + ' \n')

                #print 'after get'
                if data is not None:
                    try:
                        self.lock.acquire()
                        
                        try:
                            row = self.chan_to_row[chan]  # row in ringbuffer corresponding to this channel
                        except KeyError:
                            # print 'data source was not configured to get data on channel', chan
                            pass
                        else:
                            n_pts = len(data)
                            max_len = self.max_len

                            if n_pts > max_len:
                                data = data[-max_len:]
                                n_pts = max_len

                            idx = self.idxs[row] # for this channel, idx in ringbuffer
                            if idx + n_pts <= self.max_len:
                                self.data[row, idx:idx+n_pts] = data
                                idx = (idx + n_pts)
                                if idx == self.max_len:
                                    idx = 0
                                    if self.send_data_to_sink_manager:
                                        self.wrap_flags[row] = True
                            else: # need to write data at both end and start of buffer
                                self.data[row, idx:] = data[:max_len-idx]
                                self.data[row, :n_pts-(max_len-idx)] = data[max_len-idx:]
                                idx = n_pts-(max_len-idx)
                                if self.send_data_to_sink_manager:
                                    self.wrap_flags[row] = True
                            self.idxs[row] = idx

                        self.lock.release()

                        # Set the flag indicating that data has arrived from the source
                        self.data_has_arrived.value = 1
                    except Exception as e:
                        print(e)

                    if self.send_data_to_sink_manager:
                        self.lock.acquire()

                        # check if there is at least one column of data that
                        # has not yet been sent to the sink manager
                        if all(self.next_send_idx.value < idx + int(flag)*self.max_len for (idx, flag) in zip(self.idxs, self.wrap_flags)):
                            start_idx = self.next_send_idx.value
                            if not all(self.wrap_flags):

                                # look at minimum value of self.idxs only 
                                # among channels which have not wrapped, 
                                # in order to determine end_idx
                                end_idx = np.min([idx for (idx, flag) in zip(self.idxs, self.wrap_flags) if not flag])
                                idxs_to_send = list(range(start_idx, end_idx))
                            else:
                                min_idx = np.min(self.idxs[:])
                                idxs_to_send = list(range(start_idx, self.max_len)) + list(range(0, min_idx))
                                
                                for row in range(self.n_chan):
                                    self.wrap_flags[row] = False

                            # Old way to send data to the sink manager, one column at a time
                            # for idx in idxs_to_send:
                            #     data = np.array([tuple(self.data[:, idx])], dtype=self.send_to_sinks_dtype)
                            #     print "data shape"
                            #     print data.shape
                            #     self.sinks.send(self.name, data)

                            # # # New way to send data (in blocks) (update 1/12/2016): all columns at a time
                            #ix_ = np.ix_(np.arange(self.data.shape[0]), idxs_to_send)
                            #data = np.array(self.data[ix_], dtype=self.send_to_sinks_dtype)
                            #self.sinks.send(self.name, data)

                            #Newest way to send data to the supp hdf file, all columns at a time (1/21/2016)
                            data = np.array(list(map(tuple, self.data[:, idxs_to_send].T)), dtype = self.send_to_sinks_dtype)
                            self.supp_hdf.add_data(data)


                            self.next_send_idx.value = np.mod(idxs_to_send[-1] + 1, self.max_len)

                        self.lock.release()
            else:
                time.sleep(.001)
        
        if hasattr(self, "supp_hdf"):
            self.supp_hdf.close_data()
            print('end of supp hdf')

        system.stop()
        print(("ended datasource %r" % self.source))



    def get(self, n_pts, channels, **kwargs):
        '''
        Return the most recent n_pts of data from the specified channels.

        Parameters
        ----------
        n_pts : int
            Number of data points to read
        channels : iterable
            Channels from which to read

        Returns
        -------
        list of np.recarray objects
            Datatype of each record array is the dtype of the DataSourceSystem
        '''
        if self.status.value <= 0:
            raise Exception('\n\nError starting datasource: %s\n\n' % self.name)

        self.lock.acquire()   
   
        # these channels must be a subset of the channels passed into __init__
        n_chan = len(channels)
        data = np.zeros((n_chan, n_pts), dtype=self.source.dtype)

        if n_pts > self.max_len:
            n_pts = self.max_len
        # print "channels", channels[-1]
        for chan_num, chan in enumerate(channels):
            try:
                row = self.chan_to_row[chan]
            except KeyError:
                print(('data source was not configured to get data on channel', chan))
            else:  # executed if try clause does not raise a KeyError
                idx = self.idxs[row]
                if idx >= n_pts:  # no wrap-around required
                    data[chan_num, :] = self.data[row, idx-n_pts:idx]
                else:
                    data[chan_num, :n_pts-idx] = self.data[row, -(n_pts-idx):]
                    data[chan_num, n_pts-idx:] = self.data[row, :idx]
                self.last_read_idxs[row] = idx
        #print (data['data'])
        #self.fo2.write(str(data[0,0]['data']) + ' ' + str(len(data[0,:]['data'])) + ' ' + str(time.time()) + ' \n')
        self.lock.release()

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    def get_new(self, channels, **kwargs):
        '''
        Return the new (unread) data from the specified channels.

        Parameters
        ----------
        channels : iterable
            Channels from which to read        
        kwargs : optional kwargs 
            To be passed to self.filter, if it is listed

        Returns
        -------
        list of np.recarray objects
            Datatype of each record array is the dtype of the DataSourceSystem        
        '''
        if self.status.value <= 0:
            raise Exception('\n\nError starting datasource: %s\n\n' % self.name)

        self.lock.acquire()
        
        # these channels must be a subset of the channels passed into __init__
        n_chan = len(channels)
        data = []

        for chan in channels:
            try:
                row = self.chan_to_row[chan]
            except KeyError:
                print(('data source was not configured to get data on channel', chan))
                data.append(None)
            else:  # executed if try clause does not raise a KeyError
                idx = self.idxs[row]
                last_read_idx = self.last_read_idxs[row]
                if last_read_idx <= idx:  # no wrap-around required
                    data.append(self.data[row, last_read_idx:idx])
                else:
                    data.append(np.hstack((self.data[row, last_read_idx:], self.data[row, :idx])))
            self.last_read_idxs[row] = idx

        self.lock.release()

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    def pause(self):
        '''
        Used to toggle the 'streaming' variable in the remote "run" process 
        '''
        self.stream.set()

    def check_if_data_has_arrived(self):
        '''
        '''
        return self.data_has_arrived.value

    def stop(self):
        '''
        Set self.status.value to negative so that the while loop in self.run() terminates
        '''
        self.status.value = -1
        # self.fo2.close()
        # self.fo3.close()
    
    def __del__(self):
        '''
        Make sure the remote process stops if the Source object is destroyed
        '''
        self.stop()

    def __getattr__(self, attr):
        '''
        Try to retreive attributes from the remote DataSourceSystem if the are not found in the proximal Source object

        Parameters
        ----------
        attr : string 
            Name of attribute to retreive

        Returns
        -------
        object
            The arbitrary value associated with the named attribute, if it exists.
        '''
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        elif not attr.beginsWith("__"):
            print(("getting attribute %s" % attr))
            self.pipe.send(("getattr", (attr,), {}))
            self.cmd_event.set()
            return self.pipe.recv()
        raise AttributeError(attr)

