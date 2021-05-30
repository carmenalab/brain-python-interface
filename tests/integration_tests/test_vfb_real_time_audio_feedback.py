# TODO
# - train an LFP decoder
# - run an LFP decoder
# - CLDA
import matplotlib.pyplot as plt
import os
import pyaudio
import serial
import pandas as pd
import numpy as np
import unittest
import time
import tables


from ..features.hdf_features import SaveHDF
from ..riglib import experiment
from ..riglib import sink
from ..features import serial_port_sensor
from ..riglib import source
from ..built_in_tasks.passivetasks import TargetCaptureVFB2DWindow


class SimLFPSensor(serial_port_sensor.SerialPortSource):
    dtype = np.dtype([("ts", "f8"), ("lfp", "f8")])
    default_response = np.zeros((1,), dtype=dtype)

    START_CMD = b'a\n'
    STOP_CMD = b'b\n'

    def start(self):
        super(SimLFPSensor, self).start()
        time.sleep(1)
        print("sending start command")
        self.port.write(self.START_CMD)

    def stop(self):
        time.sleep(1)
        print("sending stop command")        
        self.port.write(self.STOP_CMD)
        super(SimLFPSensor, self).stop()
        


class SimLFPSensorFeature(object):
    def init(self):
        super().init()
        self.sensor_src = source.DataSource(SimLFPSensor, send_data_to_sink_manager=True, 
            port="/dev/cu.usbmodem1A121", baudrate=115200, name="sim_lfp")

        sink.sinks.register(self.sensor_src)

    def run(self):
        self.sensor_src.start()
        try:
            super().run()
        finally:
            self.sensor_src.stop()


class SimLFPOutput(object):
    # audio_duration = 0.1        # update amplitudes/frequencies of outputs at 10 Hz
    audio_fs = 44100       # sampling rate, Hz, must be integer    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'fps'):
            self.fps = 60
        self.audio_duration = 1.0/self.fps * 8 # TODO why is this factor necessary?
        self.samples_all = []

    def init(self):
        super().init()
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        fs = self.audio_fs
        self.audio_p = pyaudio.PyAudio()
        self.audio_stream = self.audio_p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)

        self.audio_t = np.arange(fs*self.audio_duration) * 1.0/fs
        self.phase = 0

    def _cycle(self):
        # frequency should be modulated by direction, amplitude should be modulated by speed
        vel = self.decoder[['hand_vx', 'hand_vz']]
        speed = np.linalg.norm(vel)
        direction = vel / speed
        angle = np.arctan2(direction[1], direction[0])
        if np.isnan(angle):
            angle = 0

        amp = speed / 7.5
        if amp > 0.75:
            amp = 0.75
        freq_modulation_range = 40 # Hz
        freq_base = 370
        f = angle / np.pi * freq_modulation_range/2 + freq_base
        wave_period = 1.0/f
        
        samples = amp * np.sin(2 * np.pi * f * self.audio_t + self.phase)
        self.phase += ((self.audio_duration / wave_period) % 1) * 2*np.pi
        self.samples_all.append(samples)
        self.phase = self.phase % (2 * np.pi)

        # old version, fixed frequency
        # samples = 0        
        # if self.cycle_count < 300:
        #     freq = 280
        # else:
        #     freq = 240

        # for f, amp in [(freq, 0.75)]:
        #     samples += amp * np.sin(2 * np.pi * f * self.audio_t)


        samples = samples.astype(np.float32)

        # play the audio
        self.audio_stream.write(samples)
        # self.audio_t += self.audio_duration
        super()._cycle()

    def run(self):
        super().run()
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_p.terminate() 


TestFeat = experiment.make(TargetCaptureVFB2DWindow, feats=[SaveHDF, SimLFPSensorFeature, SimLFPOutput])
# TestFeat.fps = 5
seq = TargetCaptureVFB2DWindow.centerout_2D_discrete(nblocks=2, ntargets=8) 
feat = TestFeat(seq, window_size=(480, 240))

feat.run_sync()

time.sleep(1)
hdf = tables.open_file(feat.h5file.name)
os.rename(feat.h5file.name, "test_vfb_real_time_audio_feedback.hdf")

saved_msgs = [x.decode('utf-8') for x in hdf.root.task_msgs[:]["msg"]]

lfp = hdf.root.sim_lfp[:]['lfp'][:]
ts = hdf.root.sim_lfp[:]['ts']

plt.figure()
plt.plot(hdf.root.sim_lfp[:]['lfp'])
plt.show()

plt.figure() 
plt.plot(np.log(np.abs(np.fft.fft(hdf.root.sim_lfp[:]['lfp'])))) 
plt.show()

plt.figure()
plt.specgram(lfp, Fs=1.0/(np.mean(ts) * 1e-6))
plt.show()



samples_all = np.hstack(feat.samples_all)                                                                                                        
plt.figure()
plt.plot(samples_all) 