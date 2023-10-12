
import socket
hostname = socket.gethostname()

# Defaults
rig_name = hostname
optitrack_address = None
optitrack_save_path = None
window_size = (1920, 1080)
screen_dist = 45
screen_half_height = 12
default_db = 'local'

# Rig-specific defaults
if hostname == 'pagaiisland2':
    optitrack_address = '10.155.206.1'
    optitrack_save_path = "C:/Users/Orsborn Lab/Documents"
    window_size = (2560, 1440)
    screen_dist = 28
    screen_half_height = 10.75
    default_db = 'rig1'
elif hostname == 'siberut-bmi':
    optitrack_address = '10.155.204.10'
    optitrack_save_path = "C:/Users/aolab/Documents",
    screen_dist = 28
    screen_half_height = 10.25
    default_db = 'rig2'
elif hostname == 'booted-server':
    screen_half_height = 5
    default_db = 'tablet'
elif hostname in ['moor', 'crab-eating']:
    default_db = 'rig1'

# Organize the settings
rig_settings = {
    'name': rig_name,
}

optitrack = {
    'offset': [0, -60, -30], # optitrack cm [forward, up, right]
    'scale': 1 ,# optitrack cm --> screen cm
    'address': optitrack_address,
    'save_path': optitrack_save_path,
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