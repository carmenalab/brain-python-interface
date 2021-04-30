from riglib.experiment import traits
from riglib.gpio import NIGPIO, DigitalWave
import numpy as np
import tables
import time

rig1_sync_events_ver_4 = dict(
    EXP_START               = 0x1,
    TRIAL_START             = 0x2,
    TARGET_ON               = 0x10,
    TARGET_OFF              = 0x20,
    REWARD                  = 0x30,
    PARTIAL_REWARD          = 0x31,
    HOLD_PENALTY            = 0x40,
    TIMEOUT_PENALTY         = 0x41,
    DELAY_PENALTY           = 0x42,
    OTHER_PENALTY           = 0x4f,
    CURSOR_ENTER_TARGET     = 0x50,
    CURSOR_LEAVE_TARGET     = 0x60,
    CUE                     = 0x70,
    TIME_ZERO               = 0xee,
    TRIAL_END               = 0xef,
    PAUSE                   = 0xfe,
    EXP_END                 = 0xff,    # For ease of implementation, the last event must be the highest possible value
)

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

    sync_params = dict(
        sync_protocol = 'rig1',
        sync_protocol_version = 4,
        sync_pulse_width = 0.003,
        event_sync_nidaq_mask = 0xff,
        event_sync_dch = range(16,24),
        event_sync_dict = rig1_sync_events_ver_4,
        event_sync_max_data = 0xf,
        screen_sync_nidaq_pin = 8,
        screen_sync_dch = 24,
        screen_measure_dch = [5],
        screen_measure_ach = [5],
        reward_measure_ach = [0],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_gpio = NIGPIO()
        self.sync_every_cycle = True

    def init(self, *args, **kwargs):

        # Create a record array for sync events
        if not hasattr(self, 'sinks'): # this attribute might be set in one of the other 'init' functions from other inherited classes
            from riglib import sink
            self.sinks = sink.SinkManager.get_instance()
        dtype = np.dtype([('time', 'u8'), ('event', 'S32'), ('data', 'u4'), ('code', 'u1')]) # 64-bit time (cycle number), string event, 32-bit data, 8-bit code
        self.sync_event_record = np.zeros((1,), dtype=dtype)
        self.sinks.register("sync_events", dtype)
        dtype = np.dtype([('time', 'u8'), ('timestamp', 'f8'), ('prev_tick', 'f8')])
        self.sinks.register("sync_clock", dtype)
        self.sync_clock_record = np.zeros((1,), dtype=dtype)
        super().init(*args, **kwargs)

        # Send a sync impulse to set t0
        time.sleep(self.sync_params['sync_pulse_width']*10)
        self.has_sync_event = False
        self.sync_event('TIME_ZERO', 0, immediate=True)
        self.t0 = time.perf_counter()
        time.sleep(self.sync_params['sync_pulse_width']*10)

    def sync_event(self, event_name, event_data=0, immediate=False):
        '''
        Send a sync event through NIDAQ on the next cycle, unless 'immediate' flag is set
        '''
        if self.has_sync_event:
            print("Warning: Cannot sync more than 1 event per cycle")
            print("Overwriting {} with {} event".format(self.sync_event_record['event'], event_name))

        # digital output
        code = encode_event(self.sync_params['event_sync_dict'], event_name, min(event_data, self.sync_params['event_sync_max_data']))
        self.sync_event_record['time'] = self.cycle_count
        self.sync_event_record['event'] = event_name
        self.sync_event_record['data'] = event_data
        self.sync_event_record['code'] = code
        if immediate:
            self.sinks.send("sync_events", self.sync_event_record)
            pulse = DigitalWave(self.sync_gpio, mask=0xffffff, data=self.sync_event_record['code'])
            pulse.set_pulse(self.sync_params['sync_pulse_width'], True)
            pulse.start()
            self.has_sync_event = False
        else:
            self.has_sync_event = True

    def run(self):

        # Mark the beginning and end of the experiment
        self.sync_event('EXP_START')
        try:
            super().run()            
        finally:
            time.sleep(1./self.fps) # Make sure the previous cycle is for sure over
            self.sync_event('EXP_END', event_data=0, immediate=True) # Signal the end of the experiment, even if it crashed


    def _cycle(self):
        '''
        Send a clock pulse on every cycle to the 'screen_sync_nidaq_pin'. If there are any
        sync events, also send them in the same clock cycle on the first 8 nidaq pins.
        '''
        super()._cycle()
        code = 0
        if self.sync_every_cycle:
            code = 1 << self.sync_params['screen_sync_nidaq_pin']
        if self.has_sync_event:
            self.sinks.send("sync_events", self.sync_event_record)
            code |= int(self.sync_event_record['code'])
            self.has_sync_event = False
        if code > 0:
            pulse = DigitalWave(self.sync_gpio, mask=0xffffff, data=code)
            pulse.set_pulse(self.sync_params['sync_pulse_width'], True)
            pulse.start()
        if self.sync_every_cycle:
            self.sync_clock_record['time'] = self.cycle_count
            self.sync_clock_record['timestamp'] = time.perf_counter() - self.t0
            self.sync_clock_record['prev_tick'] = self.clock.get_time()
            self.sinks.send("sync_clock", self.sync_clock_record)

    def terminate(self):
        # Reset the sync pin after experiment ends so the next experiment isn't messed up
        self.sync_gpio.write_many(0xffffff, 0)
            
    def cleanup_hdf(self):
        super().cleanup_hdf()
        if hasattr(self, "h5file"):
            h5file = tables.open_file(self.h5file.name, mode='a')
            for param in self.sync_params.keys():
                h5file.root.sync_events.attrs[param] = self.sync_params[param]
            h5file.close()
    
import copy
import pygame
from built_in_tasks.target_graphics import VirtualRectangularTarget
from riglib.stereo_opengl.window import TRANSPARENT

class ScreenSync(NIDAQSync):
    '''Adds a square in one corner that switches color with every flip.'''
    
    sync_position = {
        'TopLeft': (-1,1),
        'TopRight': (1,1),
        'BottomLeft': (-1,-1),
        'BottomRight': (1,-1)
    }
    sync_position_2D = {
        'TopLeft': (-1,-1),
        'TopRight': (1,-1),
        'BottomLeft': (-1,1),
        'BottomRight': (1,1)
    }
    sync_corner = traits.OptionsList(tuple(sync_position.keys()), desc="Position of sync square")
    sync_size = traits.Float(1, desc="Sync square size (cm)") 
    sync_color_off = traits.Tuple((0.,0.,0., 1.), desc="Sync off color (R,G,B,A)")
    sync_color_on = traits.Tuple((1.,1.,1., 1.), desc="Sync on color (R,G,B,A)")
    sync_state_duration = 1 # How long to delay the start of the experiment (seconds)

    def __init__(self, *args, **kwargs):

        # Create a new "sync" state at the beginning of the experiment
        if isinstance(self.status, dict):
            self.status["sync"] = dict(start_experiment="wait", stoppable=False)
        else:
            from riglib.fsm.fsm import StateTransitions
            self.status.states["sync"] = StateTransitions(start_experiment="wait", stoppable=False)
        self.state = "sync"

        super().__init__(*args, **kwargs)
        self.sync_state = False
        if hasattr(self, 'is_pygame_display'):
            screen_center = np.divide(self.window_size,2)
            sync_size_pix = self.sync_size * self.window_size[0] / self.screen_cm[0]
            sync_center = [sync_size_pix/2, sync_size_pix/2]
            from_center = np.multiply(self.sync_position_2D[self.sync_corner], np.subtract(screen_center, sync_center))
            top_left = screen_center + from_center - sync_center
            self.sync_rect = pygame.Rect(top_left, np.multiply(sync_center,2))
        else:
            from_center = np.multiply(self.sync_position[self.sync_corner], np.subtract(self.screen_cm, self.sync_size))
            pos = np.array([from_center[0]/2, self.screen_dist, from_center[1]/2])
            self.sync_square = VirtualRectangularTarget(target_width=self.sync_size, target_height=self.sync_size, target_color=self.sync_color_off, starting_pos=pos)
            # self.sync_square = VirtualCircularTarget(target_radius=self.sync_size, target_color=self.sync_color_off, starting_pos=pos)
            for model in self.sync_square.graphics_models:
                self.add_model(model)

    def screen_init(self):
        super().screen_init()
        if hasattr(self, 'is_pygame_display'):
            self.sync = pygame.Surface(self.window_size)
            self.sync.fill(TRANSPARENT)
            self.sync.set_colorkey(TRANSPARENT)

    def _draw_other(self):
        # For pygame display
        color = self.sync_color_on if self.sync_state else self.sync_color_off
        self.sync.fill(255*np.array(color), rect=self.sync_rect)
        self.screen.blit(self.sync, (0,0))

    def init(self):
        self.add_dtype('sync_square', bool, (1,))
        super().init()

    def _while_sync(self):
        '''
        Deliberate "startup sequence":
            1. Send a clock pulse to denote the start of the FSM loop
            2. Turn off the clock and send a single, longer, impulse
                to enable measurement of the screen latency
            3. Turn the clock back on
        '''
        
        # Turn off the clock after the first cycle is synced
        if self.cycle_count == 1:
            self.sync_every_cycle = False

        # Send an impulse to measure latency halfway through the sync state
        key_cycle = int(self.fps*self.sync_state_duration/2)
        impulse_duration = 5 # cycles, to make sure it appears on the screen
        if self.cycle_count == key_cycle:
            self.sync_every_cycle = True
        elif self.cycle_count == key_cycle + 1:
            self.sync_every_cycle = False
        elif self.cycle_count == key_cycle + impulse_duration:
            self.sync_every_cycle = True
        elif self.cycle_count == key_cycle + impulse_duration + 1:
            self.sync_every_cycle = False

    def _end_sync(self):
        self.sync_every_cycle = True

    def _test_start_experiment(self, ts):
        return ts > self.sync_state_duration

    def _cycle(self):
        super()._cycle()

        # Update the sync state
        if self.sync_every_cycle:
            self.sync_state = not self.sync_state
        self.task_data['sync_square'] = copy.deepcopy(self.sync_state)
        
        # For OpenGL display, update the graphics
        if not hasattr(self, 'is_pygame_display'):
            color = self.sync_color_on if self.sync_state else self.sync_color_off
            self.sync_square.cube.color = color