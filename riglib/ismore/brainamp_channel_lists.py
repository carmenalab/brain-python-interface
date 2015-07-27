
# 14 EMG channels
emg = [
    'AbdPolLo',
    'ExtCU',
    'ExtDig',
    'Flex',
    'PronTer',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'BackDelt',
    'Extra1',
    'Extra2',
    'Extra3',
    'Extra4',
]

# single EOG channel
eog1 = [
    'EOG',
]

# 2 EOG channels
eog2 = [
    'EOG',
    'EOG2',
]

# 32 EEG channels
eeg = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20',
    '21',
    '22',
    '23',
    '24',
    '25',
    '26',
    '27',
    '28',
    '29',
    '30',
    '31',
    '32'
]

# 16 aux channels
aux = [
    'ch1'
    'ch2'
    'ch3'
    'ch4'
    'ch5'
    'ch6'
    'ch7'
    'ch8'
    'ch9'
    'ch10'
    'ch11'
    'ch12'
    'ch13'
    'ch14'
    'ch15'
    'ch16'
]

# 35 channels total (EMG, EOG, EEG)
emg_eog2_eeg9 = emg + eog2 + eeg[0:19]

# 47 channels total (EEG, EMG, EOG)
eeg_emg_eog1 = eeg + emg + eog1

# 48 channels total (EMG, EOG, EEG)
emg_eog2_eeg = emg + eog2 + eeg
