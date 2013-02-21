'''
Converted from Plexon.h by ctypesgen on 2012-06-04
'''
from ctypes import *
# /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 410
class struct_PL_FileHeader(Structure):
    pass

struct_PL_FileHeader.__slots__ = [
    'MagicNumber',
    'Version',
    'Comment',
    'ADFrequency',
    'NumDSPChannels',
    'NumEventChannels',
    'NumSlowChannels',
    'NumPointsWave',
    'NumPointsPreThr',
    'Year',
    'Month',
    'Day',
    'Hour',
    'Minute',
    'Second',
    'FastRead',
    'WaveformFreq',
    'LastTimestamp',
    'Trodalness',
    'DataTrodalness',
    'BitsPerSpikeSample',
    'BitsPerSlowSample',
    'SpikeMaxMagnitudeMV',
    'SlowMaxMagnitudeMV',
    'SpikePreAmpGain',
    'AcquiringSoftware',
    'ProcessingSoftware',
    'Padding',
    'TSCounts',
    'WFCounts',
    'EVCounts',
]
struct_PL_FileHeader._fields_ = [
    ('MagicNumber', c_uint),
    ('Version', c_int),
    ('Comment', c_char * 128),
    ('ADFrequency', c_int),
    ('NumDSPChannels', c_int),
    ('NumEventChannels', c_int),
    ('NumSlowChannels', c_int),
    ('NumPointsWave', c_int),
    ('NumPointsPreThr', c_int),
    ('Year', c_int),
    ('Month', c_int),
    ('Day', c_int),
    ('Hour', c_int),
    ('Minute', c_int),
    ('Second', c_int),
    ('FastRead', c_int),
    ('WaveformFreq', c_int),
    ('LastTimestamp', c_double),
    ('Trodalness', c_char),
    ('DataTrodalness', c_char),
    ('BitsPerSpikeSample', c_char),
    ('BitsPerSlowSample', c_char),
    ('SpikeMaxMagnitudeMV', c_ushort),
    ('SlowMaxMagnitudeMV', c_ushort),
    ('SpikePreAmpGain', c_ushort),
    ('AcquiringSoftware', c_char * 18),
    ('ProcessingSoftware', c_char * 18),
    ('Padding', c_char * 10),
    ('TSCounts', (c_int * 5) * 130),
    ('WFCounts', (c_int * 5) * 130),
    ('EVCounts', c_int * 512),
]

# /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 472
class struct_PL_ChanHeader(Structure):
    pass

struct_PL_ChanHeader.__slots__ = [
    'Name',
    'SIGName',
    'Channel',
    'WFRate',
    'SIG',
    'Ref',
    'Gain',
    'Filter',
    'Threshold',
    'Method',
    'NUnits',
    'Template',
    'Fit',
    'SortWidth',
    'Boxes',
    'SortBeg',
    'Comment',
    'SrcId',
    'reserved',
    'ChanId',
    'Padding',
]
struct_PL_ChanHeader._fields_ = [
    ('Name', c_char * 32),
    ('SIGName', c_char * 32),
    ('Channel', c_int),
    ('WFRate', c_int),
    ('SIG', c_int),
    ('Ref', c_int),
    ('Gain', c_int),
    ('Filter', c_int),
    ('Threshold', c_int),
    ('Method', c_int),
    ('NUnits', c_int),
    ('Template', (c_short * 64) * 5),
    ('Fit', c_int * 5),
    ('SortWidth', c_int),
    ('Boxes', ((c_short * 4) * 2) * 5),
    ('SortBeg', c_int),
    ('Comment', c_char * 128),
    ('SrcId', c_ubyte),
    ('reserved', c_ubyte),
    ('ChanId', c_ushort),
    ('Padding', c_int * 10),
]

# /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 497
class struct_PL_EventHeader(Structure):
    pass

struct_PL_EventHeader.__slots__ = [
    'Name',
    'Channel',
    'Comment',
    'SrcId',
    'reserved',
    'ChanId',
    'Padding',
]
struct_PL_EventHeader._fields_ = [
    ('Name', c_char * 32),
    ('Channel', c_int),
    ('Comment', c_char * 128),
    ('SrcId', c_ubyte),
    ('reserved', c_ubyte),
    ('ChanId', c_ushort),
    ('Padding', c_int * 32),
]

# /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 508
class struct_PL_SlowChannelHeader(Structure):
    pass

struct_PL_SlowChannelHeader.__slots__ = [
    'Name',
    'Channel',
    'ADFreq',
    'Gain',
    'Enabled',
    'PreAmpGain',
    'SpikeChannel',
    'Comment',
    'SrcId',
    'reserved',
    'ChanId',
    'Padding',
]
struct_PL_SlowChannelHeader._fields_ = [
    ('Name', c_char * 32),
    ('Channel', c_int),
    ('ADFreq', c_int),
    ('Gain', c_int),
    ('Enabled', c_int),
    ('PreAmpGain', c_int),
    ('SpikeChannel', c_int),
    ('Comment', c_char * 128),
    ('SrcId', c_ubyte),
    ('reserved', c_ubyte),
    ('ChanId', c_ushort),
    ('Padding', c_int * 27),
]

# /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 533
class struct_PL_DataBlockHeader(Structure):
    pass

struct_PL_DataBlockHeader.__slots__ = [
    'Type',
    'UpperByteOf5ByteTimestamp',
    'TimeStamp',
    'Channel',
    'Unit',
    'NumberOfWaveforms',
    'NumberOfWordsInWaveform',
]
struct_PL_DataBlockHeader._fields_ = [
    ('Type', c_short),
    ('UpperByteOf5ByteTimestamp', c_ushort),
    ('TimeStamp', c_ulong),
    ('Channel', c_short),
    ('Unit', c_short),
    ('NumberOfWaveforms', c_short),
    ('NumberOfWordsInWaveform', c_short),
]


PL_FileHeader = struct_PL_FileHeader # /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 410
PL_ChanHeader = struct_PL_ChanHeader # /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 472
PL_EventHeader = struct_PL_EventHeader # /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 497
PL_SlowChannelHeader = struct_PL_SlowChannelHeader # /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 508
PL_DataBlockHeader = struct_PL_DataBlockHeader # /home/james/code/bmi3d/riglib/plexon/docs/Plexon.h: 533