import cProfile
import pstats
import zmq
import traceback
from riglib.experiment import traits
import numpy as np
import json

class Profiler():
    
    def run(self):
        pr = cProfile.Profile()
        pr.enable()
        super().run()
        pr.disable()
        with open('profile.csv', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('time')
            ps.print_stats()

class OnlineAnalysis(traits.HasTraits):
    '''
    Feature to send task data to an online analysis server.

    In the future this could be expanded to make use of the riglib.sinks interface.
    For now it is a simple UDP socket interface that sends messages about the 
    task params, state transitions, and sync events to a server for online analysis.
    '''

    online_analysis_ip = traits.String("localhost", desc="IP address of the machine running the online analysis")
    online_analysis_port = traits.Int(5000, desc="Port number for the online analysis server")
        
    def _send_online_analysis_msg(self, key, *values):
        '''
        Helper function to send messages to the online analysis server
        '''
        processed_values = []
        for v in values:
            if isinstance(v, np.ndarray):
                processed_values.append(v.tolist()) # ZMQ handles large lists easily
            elif isinstance(v, np.generic):
                processed_values.append(v.item())
            else:
                processed_values.append(v)

        try:
            # Send as a single JSON object
            # ZMQ's send_json handles the encoding/decoding for you
            self.online_analysis_sock.send_json({'key': key, 'values': processed_values})
        except Exception as e:
            print(f'ZMQ Send Error for key {key}: {e}')

    def init(self):
        '''
        Send basic experiment info to the online analysis server
        '''
        super().init()

        # Create ZMQ Context and Socket
        self.zmq_context = zmq.Context()
        self.online_analysis_sock = self.zmq_context.socket(zmq.PUSH)
        
        # Connect to the server
        addr = f"tcp://{self.online_analysis_ip}:{self.online_analysis_port}"
        self.online_analysis_sock.connect(addr)
        self.online_analysis_sock.setsockopt(zmq.SNDTIMEO, 1000) 
        self._send_online_analysis_msg('init', False) 
        
        # Send entry metadata
        try:
            self._send_online_analysis_msg('param', 'experiment_name', self.__class__.__name__)
            if hasattr(self, 'saveid'):
                self._send_online_analysis_msg('param', 'te_id', self.saveid)
            else:
                self._send_online_analysis_msg('param', 'te_id', 'None')
            if hasattr(self, 'subject_name'):
                self._send_online_analysis_msg('param', 'subject_name', self.subject_name)
        except:
            print('Problem sending entry metadata')
            traceback.print_exc()

        # Send task metadata
        for key, value in self.get_trait_values().items():
            try:
                if key in self.object_trait_names:
                    self._send_online_analysis_msg('param', key, None) # Skip objects
                    if key == 'decoder':
                        self._send_online_analysis_msg('param', 'decoder_channels', np.array(value.channels).flatten().tolist())
                        self._send_online_analysis_msg('param', 'decoder_states', value.states)
                        self._send_online_analysis_msg('param', 'decoder_bands', [(0,0)]) # TODO: How to get this?
                else:
                    self._send_online_analysis_msg('param', key, value)
            except:
                print('Problem sending task metadata')
                print(key, value)
                traceback.print_exc()

        if hasattr(self, 'sync_params'):
            for key, value in self.sync_params.items():
                try:
                    self._send_online_analysis_msg('param', key, value)
                except:
                    print('Problem sending sync param')
                    print(key, value)
                    traceback.print_exc()

        if hasattr(self, 'screen_cm'):
            self._send_online_analysis_msg('param', 'screen_cm', self.screen_cm)

        # Send init message to trigger analysis workers
        self._send_online_analysis_msg('init', True)
        print('...done!')

    def _start_wait(self):
        if hasattr(super(), '_start_wait'):
            super()._start_wait()
        if hasattr(self, 'targs') and hasattr(self, 'gen_indices'):
            for i in range(len(self.targs)):
                self._send_online_analysis_msg('target_location', self.gen_indices[i], self.targs[i])
        if hasattr(self, 'targs') and hasattr(self, 'disturbance_path'):
            self._send_online_analysis_msg('reference_signal', self.targs)
            self._send_online_analysis_msg('disturbance_signal', self.disturbance_trial, self.disturbance_path)
        
        # Eye / hand sequence trial information
        if hasattr(self, 'is_sequence'):
            self._send_online_analysis_msg('is_sequence', self.is_sequence)
            
    def _parse_next_trial(self):
        if hasattr(super(), '_parse_next_trial'):
            super()._parse_next_trial()

        # Laser conditions
        if hasattr(self, 'trial_index') and hasattr(self, 'laser_powers'):
            self._send_online_analysis_msg('laser_conditions', self.trial_index, self.laser_powers)
        if hasattr(self, 'laser_power') and hasattr(self, 'laser_index'):
            self._send_online_analysis_msg('laser_condition', self.laser_index, self.laser_power)
        
    def _cycle(self):
        '''
        Send cursor and eye position data to the online analysis server
        '''
        super()._cycle()
        self._send_online_analysis_msg('cycle_count', self.cycle_count)
        if hasattr(self, 'plant'):
            self._send_online_analysis_msg('cursor', self.plant.get_endpoint_pos())
        if hasattr(self, 'eye_pos'):
            self._send_online_analysis_msg('eye_pos', self.eye_pos)
        if hasattr(self, 'eye_diam'):
            self._send_online_analysis_msg('eye_diam', self.eye_diam)
        if hasattr(self, 'calibrated_eye_pos'):
            self._send_online_analysis_msg('calibrated_eye_pos', self.calibrated_eye_pos)

        if hasattr(self, 'task_data') and 'decoder_state' in self.task_data.dtype.names:
            self._send_online_analysis_msg('decoder_state', self.task_data['decoder_state'].flatten().tolist())
        if hasattr(self, 'task_data') and hasattr(self, 'extractor') and self.extractor.feature_type in self.task_data.dtype.names:
            self._send_online_analysis_msg('neural_features', self.task_data[self.extractor.feature_type].flatten().tolist())

    def set_state(self, condition, **kwargs):
        '''
        Send task state transitions to the online analysis server
        '''
        self._send_online_analysis_msg('state', condition)
        super().set_state(condition, **kwargs)

    def sync_event(self, event_name, event_data=0, immediate=False):
        '''
        Send sync events to the online analysis server
        '''
        if not immediate:
            self._send_online_analysis_msg('sync_event', event_name, event_data)
        super().sync_event(event_name, event_data=event_data, immediate=immediate)

class ReplayCursor():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_cursor_data = kwargs.pop('replay_cursor_data')

    def move_effector(self, *args, **kwargs):
        if self.cycle_count >= len(self.replay_cursor_data):
            self.state = None
            return
        pos = self.replay_cursor_data[self.cycle_count]
        self.task_data['manual_input'] = pos
        self.plant.set_endpoint_pos(pos)

class ReplayEye():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_eye_data = kwargs.pop('replay_eye_data')

    def _cycle(self):
        super()._cycle()
        if self.cycle_count >= len(self.replay_eye_data):
            self.state = None
            return
        pos = self.replay_eye_data[self.cycle_count]
        self.eye_pos = pos
        if 'eye' in self.task_data.dtype.names:
            self.task_data['eye'] = pos