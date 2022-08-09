# file.py
# File loading functions
import aopy
from dataclasses import dataclass
import scipy
import os
import tempfile

@dataclass
class Info():
    length: int
    units: list
    channels: list

def parse_file(filepath):
    metadata = aopy.data.load_ecube_metadata(filepath, 'Headstages')
    length = metadata['n_samples']/metadata['samplerate']
    channels = list(range(1, metadata['n_channels']+1)) # TODO incorporate the dr. map channel mapping into aopy, then use it here
    units = channels # assume multiunit exists?
    info = Info(length, units, channels)
    return info

def load_lfp(filepath, samplerate=1000):
    h5file = tempfile.NamedTemporaryFile(suffix=".hdf", delete=False)
    h5file.flush()
    h5file.close()
    tmp_filepath = h5file.name
    # tmp_filepath = '/home/pagaiisland/leo/temp_file.hdf'
    files = {
        'ecube': filepath,
    }
    aopy.preproc.proc_lfp('', files, '', tmp_filepath, overwrite=True)
    import h5py
    file = h5py.File(tmp_filepath, 'r')
    lfp_data = file['lfp_data']
    # broadband_data = aopy.data.load_ecube_data(filepath, 'Headstages')
    
    # TODO write downsampling code in aopy!!!! this is untested!!!!
    # metadata = aopy.data.load_ecube_metadata(filepath, 'Headstages')
    # bb_samples = metadata['n_samples']
    # lfp_samples = int(bb_samples/(metadata['samplerate']/samplerate))
    # lfp_data = scipy.signal.resample(broadband_data, lfp_samples, axis=0)
    return lfp_data

def load_bmi3d_cycle_times(files):
    data, metadata = aopy.preproc.parse_bmi3d("", files)
    times = data['clock']['timestamp_sync']
    return times
