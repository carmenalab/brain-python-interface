'''
Features for sending digital sync data
'''
from riglib.experiment import traits
from riglib.gpio import NIGPIO, ArduinoGPIO, DigitalWave, TeensyGPIO
import numpy as np
import tables
import time
import copy
import pygame
from built_in_tasks.target_graphics import VirtualRectangularTarget
from riglib.stereo_opengl.window import TRANSPARENT

rig1_sync_events = dict(
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
    PAUSE_START             = 0x80,
    PAUSE_END               = 0x81,
    TIME_ZERO               = 0xee,
    TRIAL_END               = 0xef,
    PAUSE                   = 0xfe,
    EXP_END                 = 0xff,    # For ease of implementation, the last event must be the highest possible value
)

hdf_sync_params = dict(
    sync_protocol = 'hdf',
    sync_protocol_version = 0,
    sync_pulse_width = 0.,
    event_sync_dict = rig1_sync_events,
    event_sync_max_data = 0xd,
    event_sync_data_shift = 0,
)

rig1_sync_params = copy.copy(hdf_sync_params)
rig1_sync_params.update(dict(
    sync_protocol = 'rig1',
    sync_protocol_version = 10,
    sync_pulse_width = 0.003,
    event_sync_mask = 0xffffff,
    event_sync_dch = range(16,24),
    screen_sync_pin = 8,
    screen_sync_dch = 24,
    screen_measure_dch = [5],
    screen_measure_ach = [5],
    reward_measure_ach = [0],
    right_eye_ach = [8, 9],
    left_eye_ach = [10, 11],
    recording_pin = 9,
    recording_dch = 25,
))

rig1_sync_params_arduino = copy.copy(rig1_sync_params)
rig1_sync_params_arduino.update(dict(
    sync_protocol = 'rig1_arduino',
    sync_protocol_version = 13,
    event_sync_mask = 0xfffffc,
    event_sync_data_shift = 2,
    event_sync_dch = range(31,39),
    screen_sync_pin = 10,
    screen_sync_dch = 39,
    recording_pin = 11,
    recording_dch = 40,

))


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

class HDFSync(traits.HasTraits):
    '''
    Sync events to the HDF file in a separate 'sync_events' dataset
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_params = hdf_sync_params
        self.sync_every_cycle = False

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
        print("Sending t0 event")
        self.sync_event('TIME_ZERO', 0, immediate=True)
        self.t0 = time.perf_counter()
        time.sleep(self.sync_params['sync_pulse_width']*10)

    def sync_event(self, event_name, event_data=0, immediate=False):
        '''
        Send a sync event on the next cycle, unless 'immediate' flag is set
        '''
        if self.has_sync_event:
            if self.sync_event_record['code'] == self.sync_params['event_sync_dict']['TRIAL_END'] and event_name == 'PAUSE_START':
                pass
            else:
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
            self.sync_code(int(self.sync_event_record['code']) << self.sync_params['event_sync_data_shift'])
            if hasattr(self, 'pulse'):
                self.pulse.join()
            self.has_sync_event = False
        else:
            self.has_sync_event = True

    def sync_code(self, code, delay=0.):
        '''
        Send a sync code through an digital io board
        '''
        pass

    def _cycle(self):
        '''
        Send a clock pulse on every cycle to the 'screen_sync_pin'. If there are any
        sync events, also send them in the same clock cycle.
        '''
        super()._cycle()
        code = 0
        if self.sync_every_cycle and 'screen_sync_pin' in self.sync_params:
            code = 1 << self.sync_params['screen_sync_pin']
        if self.has_sync_event:
            self.sinks.send("sync_events", self.sync_event_record)
            code |= int(self.sync_event_record['code']) << self.sync_params['event_sync_data_shift']
            self.has_sync_event = False
        if code > 0:
            self.sync_code(code)
        if self.sync_every_cycle:
            self.sync_clock_record['time'] = self.cycle_count
            self.sync_clock_record['timestamp'] = time.perf_counter() - self.t0
            self.sync_clock_record['prev_tick'] = self.clock.get_time()
            self.sinks.send("sync_clock", self.sync_clock_record)
            
    def cleanup_hdf(self):
        super().cleanup_hdf()
        if hasattr(self, "h5file"):
            h5file = tables.open_file(self.h5file.name, mode='a')
            for param in self.sync_params.keys():
                h5file.root.sync_events.attrs[param] = self.sync_params[param]
            h5file.close()

class NIDAQSync(HDFSync):
    '''
    Adds digital output to HDF sync to syncronize with an external recording device
    '''

    def __init__(self, *args, **kwargs):
        super(HDFSync, self).__init__(*args, **kwargs)
        self.sync_params = rig1_sync_params
        self.sync_gpio = NIGPIO()
        self.sync_every_cycle = True
        print("NIDAQ sync active")

    def sync_code(self, code, delay=0.):
        '''
        Send a sync code through GPIO
        '''
        pulse = DigitalWave(self.sync_gpio, mask=self.sync_params['event_sync_mask'], data=code)
        pulse.set_edges([0, delay, self.sync_params['sync_pulse_width']], False)
        pulse.start()
        self.pulse = pulse

    def terminate(self):
        # Reset the sync pin after experiment ends so the next experiment isn't messed up
        self.sync_gpio.write_many(self.sync_params['event_sync_mask'], 0)

    def _cycle(self):
        # Wait for the previous sync to finish before starting this cycle! 
        # Important for preserving all the clock pulses, although it leads to slowdowns in framerate
        if hasattr(self, 'pulse'):
            self.pulse.join()
        super()._cycle()
            
class ArduinoSync(NIDAQSync):
    '''
    Use an arduino microcontroller to sync instead of a NI DIO card.
    '''

    sync_gpio_port = traits.String("/dev/teensydio", desc="Port used for digital sync")
    hidden_traits = ["sync_gpio_port"]
    
    def __init__(self, *args, **kwargs):
        super(HDFSync, self).__init__(*args, **kwargs)
        self.sync_params = rig1_sync_params_arduino
        self.sync_gpio = TeensyGPIO(self.sync_gpio_port)
        self.sync_every_cycle = True


class ScreenSync(traits.HasTraits):
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
    sync_state_duration = traits.Float(2., desc="How long to delay the start of the experiment (seconds)")
    sync_state_fps = traits.Float(30., desc="Frame rate during the sync state (lower helps to measure the screen latency)")

    hidden_traits = ['sync_color_off', 'sync_color_on', 'sync_state_duration', 'sync_state_fps']

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
            pos = np.array([from_center[0]/2, 1-self.screen_dist, from_center[1]/2])
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

    def _start_sync(self):
        self._tmp_fps = copy.deepcopy(self.fps)
        self.fps = self.sync_state_fps
        # if hasattr(self, 'decoder'):
        #     self.decoder.set_call_rate(1./self.fps)

    def _end_sync(self):
        self.fps = self._tmp_fps
        # if hasattr(self, 'decoder'):
        #     self.decoder.set_call_rate(1./self.fps)
        #     print("restore update rate")

    def _test_start_experiment(self, ts):
        return ts > self.sync_state_duration

    def _cycle(self):

        # Update the sync state
        if hasattr(self, 'sync_every_cycle') and self.sync_every_cycle:
            self.sync_state = not self.sync_state
        self.task_data['sync_square'] = copy.deepcopy(self.sync_state)
        
        # For OpenGL display, update the graphics
        if not hasattr(self, 'is_pygame_display'):
            color = self.sync_color_on if self.sync_state else self.sync_color_off
            self.sync_square.cube.color = color

        super()._cycle()

class CursorAnalogOut(traits.HasTraits):
    '''
    Output cursor x and z as analog voltages. Scales cursor position by 'cursor_out_gain', then
    centers on 1.15 volts. Outputs voltages using 12 bits of resolution on two channels of a 
    teensy microcontroller.
    '''

    cursor_gpio_port = traits.String("/dev/teensyao", desc="Port used for cursor analog out")
    cursor_x_pin = traits.Int(66, desc="Pin used to output cursor x position")
    cursor_z_pin = traits.Int(67, desc="Pin used to output cursor z position")
    cursor_x_ach = traits.Int(3, desc="Analog channel used to record cursor x position")
    cursor_z_ach = traits.Int(4, desc="Analog channel used to record cursor z position")
    cursor_out_gain = traits.Float(0.1, desc="Gain to control the output voltage of cursor position")
    hidden_traits = ["cursor_gpio_port", "cursor_x_pin", "cursor_z_pin", "cursor_x_ach", "cursor_z_ach", "cursor_out_gain"]

    def __init__(self, *args, **kwargs):

        self.cursor_output_gpio = TeensyGPIO(self.cursor_gpio_port)
        super().__init__(*args, **kwargs)

    def _cycle(self):
        pos = self.plant.get_endpoint_pos()
        voltage = pos*self.cursor_out_gain # pos (cm) * gain = voltage
        
        max_voltage = 3.3
        resolution = 12
        max_int = 2**resolution - 1
        float_values = (voltage + max_voltage/2)/max_voltage # (-1.15, +1.15) becomes (0, 1)
        float_values[float_values > 1] = 1.
        float_values[float_values < 0] = 0.
        int_values = (max_int*float_values).astype(int)
        self.cursor_output_gpio.analog_write(self.cursor_x_pin, int_values[0])
        # self.cursor_output_gpio.analog_write(self.cursor_y_pin, int_values[1])
        self.cursor_output_gpio.analog_write(self.cursor_z_pin, int_values[2])

        super()._cycle()
