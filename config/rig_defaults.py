import copy
import socket
hostname = socket.gethostname()

# Defaults
rig_name = hostname
optitrack_address = None
optitrack_save_path = None
optitrack_sync_dch = 0
window_size = (1920, 1080)
screen_dist = 45
screen_half_height = 12
default_db = 'local'
reward_address = '/dev/rewardsystem'
reward_digital_pin = 12
sync_events = dict(
    EXP_START               = 0x1,
    TRIAL_START             = 0x2,
    TARGET_ON               = 0x10,
    TARGET_OFF              = 0x20,
    REWARD                  = 0x30,
    PARTIAL_REWARD          = 0x31,
    HOLD_PENALTY            = 0x40,
    TIMEOUT_PENALTY         = 0x41,
    DELAY_PENALTY           = 0x42,
    FIXATION_PENALTY        = 0x43,
    OTHER_PENALTY           = 0x4f,
    CURSOR_ENTER_TARGET     = 0x50,
    CURSOR_LEAVE_TARGET     = 0x60,
    CUE                     = 0x70,
    PAUSE_START             = 0x80,
    PAUSE_END               = 0x81,
    FIXATION                = 0x90,
    TIME_ZERO               = 0xee,
    TRIAL_END               = 0xef,
    PAUSE                   = 0xfe,
    EXP_END                 = 0xff,    # For ease of implementation, the last event must be the highest possible value
)

hdf_sync_params = dict(
    sync_protocol = 'hdf',
    sync_protocol_version = 0,
    sync_pulse_width = 0.,
    event_sync_dict = sync_events,
    event_sync_max_data = 0xd,
    event_sync_data_shift = 0,
)
nidaq_sync_params = copy.copy(hdf_sync_params)
nidaq_sync_params.update(dict(
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
rig1_sync_params_arduino = copy.copy(nidaq_sync_params)
rig1_sync_params_arduino.update(dict(
    sync_protocol = 'rig1_arduino',
    sync_protocol_version = 15,
    event_sync_mask = 0xfffffc,
    event_sync_data_shift = 2,
    event_sync_dch = range(31,39),
    screen_sync_pin = 10,
    screen_sync_dch = 39,
    recording_pin = 11,
    recording_dch = 40,

))
rig2_sync_params_arduino = copy.copy(rig1_sync_params_arduino)
rig2_sync_params_arduino.update(dict(
    sync_protocol = 'rig2',
    sync_protocol_version = 15,
    event_sync_dch = [41,42,43,44,45,48,49,50],
    screen_sync_dch = 51,
    screen_measure_dch = [6],
    screen_measure_ach = [6],
    reward_measure_ach = [1],    
    recording_dch = 52,
))
arduino_sync_params = None

# Rig-specific defaults
if hostname == 'pagaiisland2':
    optitrack_address = '10.155.206.1'
    optitrack_save_path = "C:/Users/Orsborn Lab/Documents"
    window_size = (2560, 1440)
    screen_dist = 28
    screen_half_height = 10.75
    default_db = 'rig1'
    arduino_sync_params = rig1_sync_params_arduino
elif hostname == 'siberut-bmi':
    optitrack_address = '10.155.204.10'
    optitrack_save_path = "C:/Users/aolab/Documents",
    optitrack_sync_dch = 53 # 0-index
    screen_dist = 28
    screen_half_height = 10.25
    default_db = 'rig2'
    reward_digital_pin = 2
    arduino_sync_params = rig2_sync_params_arduino
elif hostname == 'booted-server':
    screen_half_height = 5
    default_db = 'tablet'
elif hostname in ['moor', 'crab-eating']:
    default_db = 'rig1'

# Organize the settings
rig_settings = {
    'name': rig_name,
}

reward = {
    'address': reward_address,
    'digital_pin': reward_digital_pin,
}

optitrack = {
    'offset': [0, -60, -30], # optitrack cm [forward, up, right]
    'scale': 1 ,# optitrack cm --> screen cm
    'address': optitrack_address,
    'save_path': optitrack_save_path,
    'sync_dch': optitrack_sync_dch,
}

window = {
    'window_size': window_size,
    'screen_dist': screen_dist,
    'screen_half_height': screen_half_height,
}

db = {
    'secret_dbnames': [
        'rig1',
        'rig2',
        'tablet',
        'test',
    ],
    'enable_celery': False,
    'default_db': default_db,
}