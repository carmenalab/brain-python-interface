from riglib.experiment import traits
from riglib.dio.NIUSB6501.py_comedi_control import write_to_comedi
import numpy as np

rig1_sync_events_ver_0 = dict(
    EXP_START               = 0x1,
    TRIAL_START             = 0x2,
    TARGET_ON               = 0x10,
    TARGET_OFF              = 0x20,
    REWARD                  = 0x30,
    HOLD_PENALTY            = 0x40,
    TIMEOUT_PENALTY         = 0x41,
    CURSOR_ENTER_TARGET     = 0x50,
    CURSOR_LEAVE_TARGET     = 0x51,
    CUE                     = 0x52,
    TRIAL_END               = 0xef,
    PAUSE                   = 0xfe,
    EXP_END                 = 0xff,    # For ease of implementation, the last event must be the highest possible value
)

sync_protocol = rig1_sync_events_ver_0

def encode_event(dictionary, event_name, event_data):
    value = int(dictionary[event_name] + event_data)
    decoded = decode_event(dictionary, value)
    if decoded is None or decoded[0] != event_name or decoded[1] != event_data:
        raise Exception("Cannot encode " + event_name + " + " + str(event_data))
    return value

def decode_event(dictionary, value):
    ordered_list = sorted(dictionary.items(), key=lambda x: x[1])
    for i, event in enumerate(ordered_list[1:]):
        if value < event[1]:
            event_name = ordered_list[i][0]
            event_data = value - ordered_list[i][1]
            return event_name, event_data
    if value == ordered_list[-1][1]: # check last value
        return ordered_list[-1][0], 0
    return None

class NIDAQSync(traits.HasTraits):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Save the sync protocol version 
        self.sync_protocol = sync_protocol

    def init(self, *args, **kwargs):

        # Create a record array for sync events
        if not hasattr(self, 'sinks'): # this attribute might be set in one of the other 'init' functions from other inherited classes
            from riglib import sink
            self.sinks = sink.SinkManager.get_instance()
        dtype = np.dtype([('time', 'u8'), ('event', np.str), ('code', 'u1')]) # 64-bit time (cycle number), string event, 8-bit code
        self.sync_event_record = np.zeros((1,), dtype=dtype)
        self.sinks.register("sync_events", dtype)
        self.has_sync_event = False
        super().init(*args, **kwargs)
        print('done init')

    def sync_event(self, event_name, event_data=0):
        if self.has_sync_event:
            raise Exception("Cannot sync more than 1 event per cycle")

        # digital output
        code = encode_event(self.sync_protocol, event_name, event_data)
        print("Queueing sync signal " + event_name + ": " + str(event_data))
        self.sync_event_record['time'] = self.cycle_count
        self.sync_event_record['event'] = event_name
        self.sync_event_record['code'] = code
        self.has_sync_event = True

        # display sync
        if event_name == 'TRIAL_END' or \
            event_name == 'HOLD_PENALTY' or \
            event_name == 'TIMEOUT_PENALTY':
            self.sync_every_cycle = False
        elif event_name == 'TARGET_ON':
            self.sync_every_cycle = True

    def _cycle(self):
        super()._cycle()
        if self.has_sync_event:
            self.sinks.send("sync_events", self.sync_event_record)
            code = self.sync_event_record['code']
            print("Sending code: " + str(code))
            write_to_comedi(int(code), debug=True)
            self.has_sync_event = False
            

        