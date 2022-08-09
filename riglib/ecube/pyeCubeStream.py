import socket
import os
import glob
import subprocess
import time
import zmq
import numpy as np

__version__ = '0.3.1'

class eCubeStream:

    VERSTR = b"ECZQ01"

    NUM_CH_ANALOG = 32
    NUM_VCH_DIGITAL = 64
    NUM_CH_HSBANK = 640

    CONST_HS_GAIN = 1.907348633e-7
    CONST_ANALOG_GAIN = 3.0517578125e-4

    dtypedict = {
    'Headstages': np.int16,
    'AnalogPanel': np.int16,
    'DigitalPanel': np.uint64
    }

    def __init__(self, sources=None, asfloat=False, snaddress=None, ctladdress='127.0.0.1', ctlport=49686, dataport=49676, autoshutdown=True, debug=False):
        '''Initializes a pyeCubeStream object. This is the default usage of this wrapper class.'''
        self.autoshutdown = autoshutdown
        self.proc = None
        self.debug = debug
        self.retryconnect = False

        self.snaddress = snaddress

        self.__testconnection(endpoint=(ctladdress, ctlport))

        self.ctladdr = 'tcp://'+ctladdress+':'+str(ctlport)
        self.dataaddr = 'tcp://'+ctladdress+':'+str(dataport)
        self.ctx = zmq.Context.instance()
        self.ctlsock = self.ctx.socket(zmq.REQ)
        self.datasock = self.ctx.socket(zmq.SUB)

        self.ctlconnect()

        self.opts_digitalchans = None
        self.opts_digitalchanfilter = None
        self.opts_headstagefloat = None
        self.opts_analogfloat = None

        self.isstreaming = False

        if sources is not None:
            self.add(sources, asfloat)

    def __testconnection(self, endpoint):
        '''Tests the ctladdr endpoint to see if can connect; if not, try to start servernode-control'''
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            if s.connect_ex(endpoint) != 0:
                address = endpoint[0]
                ctlport = endpoint[1]
                if self.retryconnect or address not in ('127.0.0.1', 'localhost'):
                    raise RuntimeError('Cannot connect to remote address ' + str(address) + ':' + str(ctlport))
                else:
                    self.__startctlhost(endpoint)

    def __startctlhost(self, endpoint):
        '''Start servernode-control by looking for it in import path, ../ from import path, or cwd'''
        cwdir = os.getcwd()
        libdir = os.path.dirname(os.path.realpath(__file__))
        libdirp = os.path.split(libdir)[0]

        exelist = glob.glob(os.path.join(cwdir, "servernode-control*"))
        exelist.extend(glob.glob(os.path.join(libdir, "servernode-control*")))
        exelist.extend(glob.glob(os.path.join(libdirp, "servernode-control*")))

        if len(exelist) < 1:
            raise RuntimeError('Cannot connect to remote address, cannot find servernode-control')
        elif len(exelist) > 1:
            raise RuntimeError('Cannot connect to remote address, multiple servernode-control found')

        launchcmd = [exelist[0]]

        if self.snaddress is not None and isinstance(self.snaddress, str):
            launchcmd.append(self.snaddress)

        if self.debug is True:
            self.proc = subprocess.Popen(launchcmd)
        else:
            self.proc = subprocess.Popen(launchcmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(3)

        if self.proc.poll() is not None:
            raise RuntimeError('Error from servernode-control process detected')

        self.retryconnect = True
        self.__testconnection(endpoint)

    def ctlconnect(self):
        '''Connect the zmq interface'''
        if self.debug:
            print("Connecting eCubeZMQCtl to " + self.ctladdr)
        self.ctlsock.connect(self.ctladdr)

    def ctldisconnect(self):
        '''Disconnect the zmq interface'''
        if self.debug:
            print("Disconnecting eCubeZMQCtl from " + self.ctladdr)
        self.ctlsock.disconnect(self.ctladdr)

    def listavailable(self):
        '''Returns all data sources currently available for subscribing, from the upstream eCube or ServerNode.'''
        self.ctlsock.send_multipart([self.VERSTR, b"LISTAVAIL"])
        reply = self.ctlsock.recv_multipart()

        if reply[0] == b"OKAVAIL":
            availchans = np.frombuffer(reply[1], dtype='<i4')

            if self.debug is True:
                print("Channels available:")
                for hsnum, hscnt in zip(range(availchans.size), availchans):
                    if hscnt > 0:
                        print("  HS {}: {}".format(hsnum+1, hscnt))
                print("  AnalogPanel\n  DigitalPanel")
        else:
            raise ValueError("listavailable() reply invalid")
        return (availchans, self.NUM_CH_ANALOG, self.NUM_VCH_DIGITAL)

    def listadded(self):
        '''Returns all data sources currently in subscription.'''
        self.ctlsock.send_multipart([self.VERSTR, b"LISTADDED"])
        reply = self.ctlsock.recv_multipart()

        if reply[0] == b"OKADDLIST" and len(reply) == 5:

            hschans = np.ndarray((0), dtype='<u4')
            achans = np.ndarray((0), dtype='<u4')
            dchans = np.ndarray((0), dtype='<u4')

            if len(reply[1]) >= 4:
                hschanlinear = np.frombuffer(reply[1], dtype='<u4')
                hschansarr = np.asarray(list(map(
                    lambda ch_i: divmod(ch_i, self.NUM_CH_HSBANK),
                    hschanlinear)), dtype='<u4') + 1
                hschans = ((hschansarr[:, 0]).squeeze(), (hschansarr[:, 1]).squeeze())
                if self.debug is True:
                    print("Headstage channels added:")
                    print(hschans)

            if len(reply[2]) >= 4:
                achans = np.frombuffer(reply[2], dtype='<u4') + 1
                if self.debug is True:
                    print("AnalogPanel channels added:")
                    print(achans)

            if len(reply[3]) >= 4:
                dchans = np.frombuffer(reply[3], dtype='<u4') + 1
                if self.debug is True:
                    print("DigitalPanel channels added:")
                    print(dchans)

            return (hschans, achans, dchans)

        else:
            raise ValueError("listadded() reply invalid")

    def __sourcevalidate(self, source):
        '''Validate the format of a single source in .add() or .remove() operations and return readable suggestions.'''
        if type(source) is list:
            if len(source) < 1:
                raise ValueError("Source list cannot be empty")

            if all([self.__sourcevalidate(sourceitem) for sourceitem in source]):
                return True

            return False

        elif type(source) is not tuple:
            raise TypeError("Each source must be a tuple e.g. ('Headstages', 2, (1,8)), ('DigitalPanel',)")

        if len(source) < 1:
            raise ValueError("Each source must be a tuple of size 1-3")

        if type(source[0]) is not str:
            raise TypeError("The first tuple element of each source must be 'Headstages', 'AnalogPanel', or 'DigitalPanel'")

        if source[0] == "Headstages":
            if len(source) < 2 or len(source) > 3:
                raise ValueError("Source must specify a Headstage#, and optional channel range tuple ()")
            if type(source[1]) is not int or source[1] < 1 or source[1] > 10:
                raise ValueError("Headstage# must be an integer between 1 - 10")
            if len(source) == 3:
                if type(source[2]) is not tuple or len(source[2]) != 2:
                    raise TypeError("Headstage channel range must be a tuple (startch, stopch)")
                if source[2][0] < 1 or source[2][0] > self.NUM_CH_HSBANK or source[2][1] < 1 or source[2][1] > self.NUM_CH_HSBANK:
                    raise ValueError("Headstage channels must be between 1 - " + str(self.NUM_CH_HSBANK))
                if source[2][1] < source[2][0]:
                    raise ValueError("Headstage channel (startch, stopch) range invalid")
            return True

        elif source[0] == "AnalogPanel":
            if len(source) < 1 or len(source) > 2:
                raise TypeError("AnalogPanel must be a tuple ('AnalogPanel', (startch, stopch))")
            if len(source) == 2:
                if type(source[1]) is not tuple or len(source[1]) != 2:
                    raise TypeError("AnalogPanel channel range must be a tuple (startch, stopch)")
                if source[1][0] < 1 or source[1][0] > self.NUM_CH_ANALOG or source[1][1] < 1 or source[1][1] > self.NUM_CH_ANALOG:
                    raise ValueError("AnalogPanel channels must be between 1 - " + str(self.NUM_CH_ANALOG))
                if source[1][1] < source[1][0]:
                    raise ValueError("AnalogPanel channel (startch, stopch) range invalid")
            return True
        elif source[0] == "DigitalPanel":
            if len(source) < 1 or len(source) > 2:
                raise TypeError("DigitalPanel must be a tuple ('DigitalPanel', (startch, stopch))")
            if len(source) == 2:
                if type(source[1]) is not tuple or len(source[1]) != 2:
                    raise TypeError("DigitalPanel channel range must be a tuple (startch, stopch)")
                if source[1][0] < 1 or source[1][0] > self.NUM_VCH_DIGITAL or source[1][1] < 1 or source[1][1] > self.NUM_VCH_DIGITAL:
                    raise ValueError("DigitalPanel channels must be between 1 - " + str(self.NUM_VCH_DIGITAL))
                if source[1][1] < source[1][0]:
                    raise ValueError("DigitalPanel channel (startch, stopch) range invalid")
            return True

        elif source[0] == "DigitalPanelAsChans":
            if len(source) < 1 or len(source) > 2:
                raise TypeError("DigitalPanelAsChans must be a tuple ('DigitalPanelAsChans', (startch, stopch))")
            if len(source) == 2:
                if type(source[1]) is not tuple or len(source[1]) != 2:
                    raise TypeError("DigitalPanelAsChans channel range must be a tuple (startch, stopch)")
                if source[1][0] < 1 or source[1][0] > self.NUM_VCH_DIGITAL or source[1][1] < 1 or source[1][1] > self.NUM_VCH_DIGITAL:
                    raise ValueError("DigitalPanelAsChans channels must be between 1 - " + str(self.NUM_VCH_DIGITAL))
                if source[1][1] < source[1][0]:
                    raise ValueError("DigitalPanelAsChans channel (startch, stopch) range invalid")
            return True
        else:
            print(source)
            raise ValueError("Unknown source type")

    def add(self, sources, asfloat=False):
        '''Adds data sources to the current subscription. Iterate through if source is a list.'''
        if not self.__sourcevalidate(sources):
            raise ValueError("Invalid sources")

        if type(sources) is list:
            for source in sources:
                self.__addsingle(source, asfloat)
        else:
            self.__addsingle(sources, asfloat)

    def __addsingle(self, source, asfloat=False):
        '''Called by .add() to actually add sources.'''
        if source[0] == "Headstages":
            if len(source) == 2:
                self.ctlsock.send_multipart([self.VERSTR, b"ADD", b"H",
                    source[1].to_bytes(4, 'little')])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKADD":
                    raise ValueError(reply[0].decode('utf-8'))

            elif len(source) == 3:
                self.ctlsock.send_multipart(
                    [self.VERSTR, b"ADD", b"H",
                    source[1].to_bytes(4, 'little'),
                    source[2][0].to_bytes(4, 'little'),
                    source[2][1].to_bytes(4, 'little')])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKADD":
                    raise ValueError(reply[0].decode('utf-8'))
            else:
                raise ValueError("Unknown Headstage add command")

            if asfloat is True:
                self.opts_headstagefloat = True

        elif source[0] == "AnalogPanel":
            if len(source) == 1:
                self.ctlsock.send_multipart([self.VERSTR, b"ADD", b"A"])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKADD":
                    raise ValueError(reply[0].decode('utf-8'))

            elif len(source) == 2:
                self.ctlsock.send_multipart(
                    [self.VERSTR, b"ADD", b"A",
                    source[1][0].to_bytes(4, 'little'),
                    source[1][1].to_bytes(4, 'little')])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKADD":
                    raise ValueError(reply[0].decode('utf-8'))
            else:
                raise ValueError("Unknown AnalogPanel add command")

            if asfloat is True:
                self.opts_analogfloat = True

        elif source[0] == "DigitalPanel" or source[0] == "DigitalPanelAsChans":
            if (self.opts_digitalchans is not None) and ((source[0] == "DigitalPanel" and self.opts_digitalchans is not False) or (source[0] == "DigitalPanelAsChans" and self.opts_digitalchans is not True)):
                raise RuntimeError("DigitalPanel already subscribed. Please remove before changing subscription type.")

            if len(source) == 1:
                self.ctlsock.send_multipart([self.VERSTR, b"ADD", b"D"])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKADD":
                    raise ValueError(reply[0].decode('utf-8'))

            elif len(source) == 2:
                self.ctlsock.send_multipart(
                    [self.VERSTR, b"ADD", b"D",
                    source[1][0].to_bytes(4, 'little'),
                    source[1][1].to_bytes(4, 'little')])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKADD":
                    raise ValueError(reply[0].decode('utf-8'))
            else:
                raise ValueError("Unknown DigitalPanel add command")

            if source[0] == "DigitalPanelAsChans":
                self.opts_digitalchans = True

                if len(source) == 1:
                    self.opts_digitalchanfilter = None
                if len(source) == 2:
                    if self.opts_digitalchanfilter is None:
                        self.opts_digitalchanfilter = list(range(source[1][0]-1, source[1][1]))
                    elif type(self.opts_digitalchanfilter) is list:
                        self.opts_digitalchanfilter = list(set(self.opts_digitalchanfilter).union(set(range(source[1][0]-1, source[1][1]))))
                    else:
                        raise TypeError("Digital channels python API is the wrong type")
                else:
                    raise ValueError("Unknown DigitalPanelAsChans add command")
            else:
                self.opts_digitalchans = False

        else:
            raise ValueError("Unknown add command")

    def remove(self, sources):
        '''Removes data sources from the current subscription. Iterate through if source is a list.'''
        if not self.__sourcevalidate(sources):
            raise ValueError("Invalid sources")

        if type(sources) is list:
            for source in sources:
                self.__removesingle(source)
        else:
            self.__removesingle(sources)

    def __removesingle(self, source):
        '''Called by .remove() to actually remove sources from subscription.'''
        if source[0] == "Headstages":
            if len(source) == 2:
                self.ctlsock.send_multipart([self.VERSTR, b"REMOVE", b"H",
                    source[1].to_bytes(4, 'little')])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKREMOVE":
                    raise ValueError(reply[0].decode('utf-8'))
                self.opts_headstagefloat = None

            elif len(source) == 3:
                raise ValueError("Removal operations on whole-headstage only, e.g. ('Headstages', 2)")

            else:
                raise ValueError("Unknown headstage remove command")

        elif source[0] == "AnalogPanel":
            if len(source) == 1:
                self.ctlsock.send_multipart([self.VERSTR, b"REMOVE", b"A"])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKREMOVE":
                    raise ValueError(reply[0].decode('utf-8'))
                self.opts_analogfloat = None

            elif len(source) == 2:
                raise ValueError("Removal operations on whole-AnalogPanel only, e.g. ('AnalogPanel',)")

            else:
                raise ValueError("Unknown AnalogPanel remove command")

        elif source[0] == "DigitalPanel" or source[0] == "DigitalPanelAsChans":
            if len(source) == 1:
                self.ctlsock.send_multipart([self.VERSTR, b"REMOVE", b"D"])
                reply = self.ctlsock.recv_multipart()
                if len(reply) == 1 and reply[0] != b"OKREMOVE":
                    raise ValueError(reply[0].decode('utf-8'))
                self.opts_digitalchans = None
                self.opts_digitalchanfilter = None

            elif len(source) == 2:
                raise ValueError("Removal operations on whole-DigitalPanel only, e.g. ('DigitalPanel',)")

            else:
                raise ValueError("Unknown DigitalPanel remove command")

        else:
            raise ValueError("Unknown remove command")

    def __closesession(self):
        self.ctlsock.send_multipart([self.VERSTR, b"CLOSESESSION"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKCLOSE":
            raise RuntimeError(reply[0].decode('utf-8'))

    def remotesave(self, sessionname):
        '''Start a remote data recording on the host running servernode software, based on the currently subscribed set of channels. The recordings will be saved into a newly created directory matching the method parameter sessionname.'''
        if self.__hassessions() is not True:
            raise RuntimeError("Upstream connection is not ServerNode and cannot perform remote saving")

        if type(sessionname) is not str:
            raise RuntimeError("sessionname must be a valid string")
        sessionbytes = sessionname.encode('utf-8')

        self.ctlsock.send_multipart([self.VERSTR, b"CREATESESSION", sessionbytes])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKCREATE":
            raise RuntimeError(reply[0].decode('utf-8'))

        self.ctlsock.send_multipart([self.VERSTR, b"RECORDSESSION", b"START"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKSTART":
            raise RuntimeError(reply[0].decode('utf-8'))

        self.__closesession()

    def listremotesessions(self):
        '''Lists all currently created remote sessions running on the host running servernode software.'''
        if self.__hassessions() is not True:
            raise RuntimeError("Upstream connection is not ServerNode and cannot perform remote saving")

        self.ctlsock.send_multipart([self.VERSTR, b"LISTSESSIONS"])
        reply = self.ctlsock.recv_multipart()

        if reply[0] == b"OKLIST":
            if self.debug is True:
                print("Remove saving sessions open:")
                for sess in reply[1:]:
                    print("  " + sess.decode('utf-8'))
            return [sess.decode('utf-8') for sess in reply[1:]]
        else:
            raise ValueError("listremotesessions() reply invalid")

    def remotestopsave(self, sessionname):
        '''Stop an active remote data recording on the host running servernode software.'''
        if self.__hassessions() is not True:
            raise RuntimeError("Upstream connection is not ServerNode and cannot perform remote saving")

        if type(sessionname) is not str:
            raise RuntimeError("sessionname must be a valid string")
        sessionbytes = sessionname.encode('utf-8')

        self.ctlsock.send_multipart([self.VERSTR, b"OPENSESSION", sessionbytes])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKOPEN":
            raise RuntimeError(reply[0].decode('utf-8'))

        self.ctlsock.send_multipart([self.VERSTR, b"RECORDSESSION", b"STOP"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKSTOP":
            raise RuntimeError(reply[0].decode('utf-8'))

        self.ctlsock.send_multipart([self.VERSTR, b"DESTROYSESSION"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKDESTROY":
            raise RuntimeError(reply[0].decode('utf-8'))

    def start(self):
        '''Starts the concurrent streaming of subscribed data sources, and makes available obtaining streaming data via .get().'''
        self.datasock.connect(self.dataaddr)
        self.datasock.setsockopt_string(zmq.SUBSCRIBE, '')

        self.ctlsock.send_multipart([self.VERSTR, b"START"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKSTART":
            raise RuntimeError(reply[0].decode('utf-8'))

        self.isstreaming = True
        return True

    def get(self):
        '''Obtains the latest available data packet from the stream.'''
        if self.isstreaming is not True:
            raise RuntimeError("get() must be called while streaming by issuing start()")

        datamsg = self.datasock.recv_multipart()
        samples = int.from_bytes(datamsg[2], byteorder='little', signed=True)

        sourcetype = datamsg[0].decode('utf-8')
        if sourcetype not in self.dtypedict.keys():
            raise TypeError('Invalid sourcetype ' + sourcetype + ' detected.')

        if sourcetype == 'DigitalPanel':
            channels = 1
        else:
            channels = int.from_bytes(datamsg[3], byteorder='little', signed=False)

        if len(datamsg[4]) != channels * samples * np.dtype(self.dtypedict[sourcetype]).itemsize:
            raise ValueError("Error: data packet size {} did not match {} samples of {} channels".format(
                len(datamsg[4]), samples, channels))

        timestamp = int.from_bytes(datamsg[1], byteorder='little', signed=False)

        if sourcetype == 'DigitalPanel' and self.opts_digitalchans is True:
            data = np.frombuffer(datamsg[4], dtype='<u1')
            data = np.unpackbits(data, bitorder='little')
            channels = self.NUM_VCH_DIGITAL
        else:
            data = np.frombuffer(datamsg[4], dtype=self.dtypedict[sourcetype])

        data = data.reshape((samples, channels))

        if self.opts_headstagefloat is True and sourcetype == 'Headstages':
            data = data.astype(np.single) * self.CONST_HS_GAIN

        if self.opts_analogfloat is True and sourcetype == 'AnalogPanel':
            data = data.astype(np.single) * self.CONST_ANALOG_GAIN

        if sourcetype == 'DigitalPanel' and self.opts_digitalchans is True and self.opts_digitalchanfilter is not None:
            data = data[:, self.opts_digitalchanfilter]
            sourcetype = 'DigitalPanelAsChans'

        return (timestamp, sourcetype, data)

    def stop(self):
        '''Stops the concurrent streaming of subscribed data sources.'''
        self.ctlsock.send_multipart([self.VERSTR, b"STOP"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKSTOP":
            raise ValueError(reply[0].decode('utf-8'))

        self.datasock.disconnect(self.dataaddr)

        self.isstreaming = False
        return True

    def resetheadstage(self, hsnum):
        '''Sends headstage reset command.'''

        if not isinstance(hsnum, int) or hsnum < 1 or hsnum > 10:
            raise ValueError("Headstage# must be an integer between 1 - 10")

        self.ctlsock.send_multipart([self.VERSTR, b"RESETHS",
            hsnum.to_bytes(4, 'little')])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] != b"OKRESETHS":
            raise ValueError(reply[0].decode('utf-8'))

        return True

    def __hassessions(self):
        self.ctlsock.send_multipart([self.VERSTR, b"HASSESSIONS"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] == b"OKSESSIONS":
            return True
        return False

    def __isstreaming(self):
        self.ctlsock.send_multipart([self.VERSTR, b"ISSTREAMING"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] == b"OKSTREAMING":
            return True
        return False

    def __issession(self):
        self.ctlsock.send_multipart([self.VERSTR, b"ISSESSION"])
        reply = self.ctlsock.recv_multipart()
        if len(reply) == 1 and reply[0] == b"OKINSESSION":
            return True
        return False

    def __del__(self):
        '''Cleans up on shutdown.'''
        if self.autoshutdown:

            if hasattr(self, 'ctlsock'):
                if self.__isstreaming():
                    self.stop()

                if self.__issession():
                    self.__closesession()

                self.ctldisconnect()

            if self.proc is not None:
                try:
                    outs, errs = self.proc.communicate(input="quit\n", timeout=2)
                except:
                    self.proc.kill()
                    outs, errs = self.proc.communicate()
