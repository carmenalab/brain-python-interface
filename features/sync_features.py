
from riglib.experiment import traits

class Rig1SyncEventProtocol():
    '''
    Events are encoded between 1 and 255 to send over 8 digital lines to the recording system. 
    '''
    version = '0'

    # Base events
    EXP_START               = 0x1
    TRIAL_START             = 0x2
    TARGET_ON               = 0x10
    TARGET_OFF              = 0x20
    REWARD                  = 0x30
    HOLD_PENALTY            = 0x40
    TIMEOUT_PENALTY         = 0x41
    CURSOR_ENTER_TARGET     = 0x50
    CURSOR_LEAVE_TARGET     = 0x51
    CUE                     = 0x52
    TRIAL_END               = 0xef
    PAUSE                   = 0xfe
    EXP_END                 = 0xff

    def __init__(self, event, data):
        if type(event) is str:
            self.name = event
            self.base_code = getattr(self, event)
        else:
            tmp = self.decode(event) # lookup the string name
            self.name = tmp.name
            self.base_code = event
        self.data = data

    def __str__(self):
        return "%s: %d + %d" % (self.name, self.base_code, self.data)
    
    def __repr__(self):
        return self.__class__.__module__ + "." + self.__class__.__name__ + " " + self.__str__()

    def encode(self):
        value = int(self.base_code + self.data)
        tmp = self.decode(value)
        if tmp is None or tmp.base_code != self.base_code:
            raise Exception("Cannot encode " + str(self))
        return value

    @classmethod
    def decode(cls, value):
        data = 0

        # Only some events have data
        if value >= cls.TRIAL_START and value < cls.TARGET_ON:
            event = 'TRIAL_START'
            data = value - cls.TRIAL_START
        elif value >= cls.TARGET_ON and value < cls.TARGET_OFF:
            event = 'TARGET_ON'
            data = value - cls.TARGET_ON
        elif value >= cls.TARGET_OFF and value < cls.REWARD:
            event = 'TARGET_OFF'
            data = value - cls.TARGET_OFF
        else:
            events = [(key, value) for key, value in dict(vars(cls)).items() if not key.startswith("__") and not key in ['version', 'encode', 'decode']]
            reverse = dict((value, str(key)) for key, value in events)
            try:
                event = reverse[value]
            except KeyError:
                print("Cannot decode value " + str(value))
                return
        return cls(event, data)

sync_events = Rig1SyncEventProtocol

class NIDAQSync(traits.HasTraits):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_queue = []

        # Save the sync protocol version 
        self.sync_protocol = sync_events
        self.sync_protocol_version = sync_events.version # <-- TODO does this get saved anywhere?

    def sync_event(self, event_name, event_data=0):
    
        # digital output
        event = sync_events(event_name, event_data)
        print("Queueing sync signal " + str(event))
        self.sync_queue.append(sync_events(event_name, event_data))

        # display sync
        if event_name == sync_events.TRIAL_END or \
            event_name == sync_events.HOLD_PENALTY or \
            event_name == sync_events.TIMEOUT_PENALTY:
            self.sync_every_cycle = False
        elif event_name == sync_events.TARGET_ON:
            self.sync_every_cycle = True

    def _cycle(self):
        super()._cycle()
        while len(self.sync_queue) > 0:
            event = self.sync_queue.pop(0)
            code = event.encode()
            print("Sending code: " + str(code))
            # TODO We should save each event to the hdf file as well along with the frame number

        