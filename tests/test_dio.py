import numpy as np
import time
from riglib.gpio import ArduinoGPIO, DigitalWave, TeensyGPIO
from riglib import source
from riglib.ecube import Digital, LFP
from features.sync_features import ArduinoSync
from config.rig_defaults import rig2_sync_params_arduino
from config.rig_defaults import reward
import aopy

import unittest

sync_params = rig2_sync_params_arduino # rig1_sync_params_arduino

class TestDIO(unittest.TestCase):

    # def test_reward_out(self):
    #     dio = ArduinoGPIO(reward['address'])
    #     time.sleep(5)
        
    #     for pin in range(2,14):
    #         print(pin)
    #         # Send a bunch of pulses
    #         pulse_width = 0.2
    #         pulse_interval = 0.2
    #         duration = 1
    #         t0 = time.perf_counter()
    #         while time.perf_counter() - t0 < duration:
    #             dio.write(pin, 1)
    #             t1 = time.perf_counter()
    #             while time.perf_counter() - t1 < pulse_width:
    #                 time.sleep(0)
    #             dio.write(pin, 0)
    #             t1 = time.perf_counter()
    #             while time.perf_counter() - t1 < pulse_interval:
    #                 time.sleep(0)

    # @unittest.skip("")
    def test_dio_speed_direct(self):
        print("Testing direct connection to DIO")

        # Connect to the DIO
        mask = sync_params['event_sync_mask']
        print(f'using mask: {mask}')
        shift = sync_params['event_sync_data_shift']
        dio = TeensyGPIO('/dev/ttyACM0', baudrate=115200)
        dio.write_many(mask, 0)

        # Record data from the digital sync channels
        channels = [ch+1 for ch in sync_params['event_sync_dch']]
        ds = source.MultiChanDataSource(Digital, channels=channels, bufferlen=10)
        ds.start()
        time.sleep(1)
        
        # Send a bunch of pulses
        pulse_width = 0.003
        pulse_data = 255
        pulse_interval = 0.003
        duration = 5
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            dio.write_many(mask, pulse_data << shift)
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < pulse_width:
                time.sleep(0)
            dio.write_many(mask, 0)
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < pulse_interval:
                time.sleep(0)

        # Record the data
        time.sleep(1)
        data = ds.get_new(channels)
        ds.stop()
        ecube_dig_channels = np.squeeze(data).T
        nch = len(channels)
        digital_samplerate = 25000

        # Make sure all the pulses have the right data
        digital_data = np.squeeze(np.packbits(ecube_dig_channels, bitorder='little').view(np.dtype('<u8')))
        # event_bit_mask = aopy.utils.convert_channels_to_mask(channels)
        ecube_sync_data = aopy.utils.mask_and_shift(digital_data, 0xff)
        raw_times, raw_events = aopy.utils.detect_edges(ecube_sync_data, digital_samplerate, rising=True, falling=False)
        print(f"Recorded {len(raw_times)} events\n---------------------------")
        np.testing.assert_allclose(raw_events, pulse_data)

        # Separate individual pulses into "ok", "too_short" and "too_long"
        pulses_ok = []
        pulses_too_short = []
        pulses_too_long = []

        for ch in range(nch):
            raw_times, raw_events = aopy.utils.detect_edges(ecube_dig_channels[:,ch], digital_samplerate, rising=True, falling=True)
            
            widths = raw_times[1::2]-raw_times[::2]
            
            width_too_short = widths < 0.002
            width_too_long = widths > 0.004
            width_ok = (~width_too_short) & (~width_too_long)
            
            # Save the timestamps of the events
            pulses_too_short.append(raw_times[::2][width_too_short])
            pulses_too_long.append(raw_times[::2][width_too_long])
            pulses_ok.append(raw_times[::2][width_ok])

        # Collect all the channels together
        width_too_short = np.concatenate(pulses_too_short, axis=0)
        print(f"too short: {np.count_nonzero(width_too_short)} pulses")
        width_too_long = np.concatenate(pulses_too_long, axis=0)
        print(f"too long: {np.count_nonzero(width_too_long)} pulses")
        width_ok = np.concatenate(pulses_ok, axis=0)
        print(f"ok: {np.count_nonzero(width_ok)} pulses")

    @unittest.skip("")
    def test_dio_speed_multithreaded(self):
        print("Testing DIO via ArduinoSync()")

        # Use the sync feature to connect to the DIO
        dio = ArduinoSync()

        # Record data from the digital sync channels
        channels = dio.sync_params['event_sync_dch']
        ds = source.MultiChanDataSource(Digital, channels=channels, bufferlen=10)
        ds.start()
        time.sleep(1)
        
        # Send a bunch of pulses
        pulse_width = 0.003
        pulse_data = 255
        pulse_interval = 0.003
        duration = 5
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            dio.sync_code(int(pulse_data) << dio.sync_params['event_sync_data_shift'])
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < pulse_width + pulse_interval:
                time.sleep(0)

        # Record the data
        time.sleep(1)
        data = ds.get_new(channels)
        ds.stop()
        ecube_dig_channels = np.array(data).T
        nch = len(channels)
        digital_samplerate = 25000

        # Make sure all the pulses have the right data
        digital_data = np.squeeze(np.packbits(ecube_dig_channels, bitorder='little').view(np.dtype('<u8')))
        event_bit_mask = aopy.utils.convert_channels_to_mask(channels)
        ecube_sync_data = aopy.utils.mask_and_shift(digital_data, event_bit_mask)
        raw_times, raw_events = aopy.utils.detect_edges(ecube_sync_data, digital_samplerate, rising=True, falling=False)
        print(f"Recorded {len(raw_times)} events\n---------------------------")
        np.testing.assert_allclose(raw_events, pulse_data)

        # Separate individual pulses into "ok", "too_short" and "too_long"
        pulses_ok = []
        pulses_too_short = []
        pulses_too_long = []

        for ch in range(nch):
            raw_times, raw_events = aopy.utils.detect_edges(ecube_dig_channels[:,ch], digital_samplerate, rising=True, falling=True)
            
            widths = raw_times[1::2]-raw_times[::2]
            
            width_too_short = widths < 0.002
            width_too_long = widths > 0.004
            width_ok = (~width_too_short) & (~width_too_long)
            
            # Save the timestamps of the events
            pulses_too_short.append(raw_times[::2][width_too_short])
            pulses_too_long.append(raw_times[::2][width_too_long])
            pulses_ok.append(raw_times[::2][width_ok])

        # Collect all the channels together
        width_too_short = np.concatenate(pulses_too_short, axis=0)
        print(f"too short: {np.count_nonzero(width_too_short)} pulses")
        width_too_long = np.concatenate(pulses_too_long, axis=0)
        print(f"too long: {np.count_nonzero(width_too_long)} pulses")
        width_ok = np.concatenate(pulses_ok, axis=0)
        print(f"ok: {np.count_nonzero(width_ok)} pulses")

if __name__ == '__main__':
    unittest.main()