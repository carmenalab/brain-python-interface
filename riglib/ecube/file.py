# file.py
# File loading functions
from .dataset import Dataset
from dataclasses import dataclass

@dataclass
class Info():
    length: int
    units: list
    channels: list

def parse_file(filepath):
    n_channels = 0
    n_samples = 0
    dat = Dataset(filepath)
    recordings = dat.listrecordings()
    for r in recordings: # r: (data_source, n_channels, n_samples)
        if 'Headstages' in r[0]:
            n_samples += r[2]  
            n_channels = r[1]
    samplerate = dat.samplerate
    length = n_samples/samplerate
    units = []
    channels = list(range(n_channels)) # TODO incorporate the dr. map channel mapping
    info = Info(length, units, channels)
    return info

