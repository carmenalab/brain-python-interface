'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
import numpy as np
import os
import tables
import time
import subprocess
import tempfile
from OpenGL.GL import GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE
import pygame

from riglib.experiment import traits, Experiment
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.bmi import Decoder, BMILoop, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.primitives import TexPlane
from riglib.stereo_opengl.textures import Texture
from riglib.stereo_opengl.window import Window, WindowDispl2D, Window2D
from config.rig_defaults import window as window_defaults

from built_in_tasks.manualcontrolmultitasks import ScreenTargetCapture
from built_in_tasks.bmimultitasks import BMIControlMulti

from .target_graphics import *

from .bmimultitasks import BMIControlMultiEyeConstrained

bmi_ssm_options = ['Endpt2D', 'Tentacle', 'Joint2L']

class EndPostureFeedbackController(BMILoop, traits.HasTraits):
    ssm_type_options = bmi_ssm_options
    ssm_type = traits.OptionsList(*bmi_ssm_options, bmi3d_input_options=bmi_ssm_options)
    decoder_update_rate = traits.Float(60, desc="Assist feedback rate (Hz)")
    
    static_states = ['wait', 'reward', 'fixation_penalty', 'pause'] # states in which the decoder is not run

    def load_decoder(self):
        self.ssm = StateSpaceEndptVel2D()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=1./self.decoder_update_rate)
        self.decoder.n_features = 1
        self.decoder.is_null_decoder = True
        

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()


class TargetCaptureVisualFeedback(EndPostureFeedbackController, BMIControlMulti):
    assist_level = (1, 1)
    is_bmi_seed = True

    def move_effector(self, *args, **kwargs):
        pass

class BMIControlVisualFeedbackEyeConstrained(BMIControlMultiEyeConstrained):
    '''What is the plan here? I'm going to slowly transition the visual feedback task to this so I can dynamically add
    either the automatic control or decoder control'''
    blink_time_threshold = traits.Float(0.1, desc="The amount of time in seconds that the eyes can be closed before triggering a fixation break, measured by eye_diam=0")
    assist_level = (1, 1)
    is_bmi_seed = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.most_recent_open_eye = 0

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty", enter_target="hold", start_pause="pause", fixation_break="fixation_penalty"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay", fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def _start_target(self):
        self.plant.set_visibility(True)
        #target capture 
        self.target_index += 1

        #screen target capture: Modified
        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        target = self.targets[self.target_index % 2]
        if self.target_index == 0:
            self.sync_event('TARGET_ON', self.gen_indices[self.target_index])
        self.target_location = self.targs[self.target_index] # save for BMILoop

    def _start_fixation_penalty(self):
        self.plant.set_visibility(False)
        super()._start_fixation_penalty()
        self.decoder.filt.state.mean = self.init_decoder_mean.copy()

    def _start_wait(self):
        super()._start_wait()
        #self.plant_visible = False
        self.plant.set_visibility(True)

    def _end_targ_transition(self):
        self.plant.set_visibility(False)
        super()._end_targ_transition()

    def _start_pause(self):
        super()._start_pause()
        self.plant.set_visibility(False)

    def _end_pause(self):
        super()._end_pause()
        # Reset on any target transition away from the last target
        self.decoder.filt.state.mean = self.init_decoder_mean.copy()
        self.hdf.sendMsg("reset")
        
        self.plant.set_visibility(True)

    def _test_fixation_break(self,time_in_state):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        '''  

        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        #Make flag that tracks the last non-zero eye diameter and check that it has occured in the last 100ms
        #First logic check: Check to see if the eye is open. If open, reset flag to 0   
        eye_within_fixation_buffer = (eye_d > self.target_radius + self.fixation_radius_buffer)
        if self.keyboard_control:
            return eye_within_fixation_buffer
        elif np.any(self.eye_diam != 0):
            self.most_recent_open_eye = 0
        elif self.most_recent_open_eye == 0: #Additionally check if this if the first cycle of 'eyes closed'. If not the first cycle, check how long since the flag was set and trigger a fixatio nfailure if longer than 100ms.
            self.most_recent_open_eye=self.get_time()
        elif (self.get_time()-self.most_recent_open_eye) > self.blink_time_threshold:
            self.most_recent_open_eye = 0
            return True            
    
        #Finally check if the eye location is within the target + buffer
        return eye_within_fixation_buffer
    
    def _test_start_trial(self, time_in_state):
        #Check that the eye position is on the center target
        #return True #super()._test_start_trial
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[0,[0,2]]) #target index is zero, this is only applyied during the wait period before the center target comes on
        
        blink = self.keyboard_control | np.any(self.eye_diam!=0)

        value = (eye_d < self.target_radius + self.fixation_radius_buffer) & blink
        return value

class TargetCaptureVisualFeedbackEyeConstrained(EndPostureFeedbackController, BMIControlVisualFeedbackEyeConstrained):
    '''def move_effector(self):
        #Follow up on this move_effector. I think it shouldn't be here?
        pass'''
    pass





class TargetCaptureVFB2DWindow(TargetCaptureVisualFeedback, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(TargetCaptureVFB2DWindow, self).__init__(*args, **kwargs)
        self.assist_level = (1, 1)

    def _start_wait(self):
        self.wait_time = 0.
        super(TargetCaptureVFB2DWindow, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    @classmethod
    def get_desc(cls, params, report):
        if isinstance(report, list) and len(report) > 0:
            duration = report[-1][-1] - report[0][-1]
            reward_count = 0
            for item in report:
                if item[0] == "reward":
                    reward_count += 1
            return "{} rewarded trials in {} min".format(reward_count, int(np.ceil(duration / 60)))
        elif isinstance(report, dict):
            duration = report['runtime'] / 60
            reward_count = report['n_success_trials']
            return "{} rewarded trials in {} min".format(reward_count, int(np.ceil(duration / 60)))
        else:
            return "No trials"

from .target_graphics import target_colors

class TargetCaptureReplay(ScreenTargetCapture):
    '''
    Reads the frame-by-frame cursor and trial-by-trial target positions from a saved
    HDF file to display an exact copy of a previous experiment. 
    Doesn't really work, do not recommend using this.
    '''

    hdf_filepath = traits.String("", desc="Filepath of hdf file to replay")

    exclude_parent_traits = list(set(ScreenTargetCapture.class_traits().keys()) - \
        set(['window_size', 'fullscreen']))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = time.perf_counter()
        with tables.open_file(self.hdf_filepath, 'r') as f:
            task = f.root.task.read()
            state = f.root.task_msgs.read()
            trial = f.root.trials.read()
            params = f.root.task.attrs._f_list("user")
            self.task_meta = {k : getattr(f.root.task.attrs, k) for k in params}
        self.replay_state = state
        self.replay_task = task
        self.replay_trial = trial
        for k, v in self.task_meta.items():
            if k in self.exclude_parent_traits:
                print("setting {} to {}".forstart_trialmat(k, v))
                setattr(self, k, v)

        # Have to additionally reset the targets since they are created in super().__init__()
        target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        self.targets = [target1, target2]

    def _test_start_trial(self, time_in_state):
        '''Wait for the state change in the HDF file in case there is autostart enabled'''
        print('testing_start_trial')
        trials = self.replay_state[self.replay_state['msg'] == b'target']
        upcoming_trials = [t['time']-1 for t in trials if self.replay_task[t['time']]['trial'] >= self.calc_trial_num()]
        return (np.array(upcoming_trials) <= self.cycle_count).any()

    def _parse_next_trial(self):
        '''Ignore the generator'''
        self.targs = []
        self.gen_indices = []
        trial_num = self.calc_trial_num()
        for trial in self.replay_trial:
            if trial['trial'] == trial_num:
                self.targs.append(trial['target'])
                self.gen_indices.append(trial['index'])

    def _cycle(self):
        '''Have to fudge the cycle_count a bit in case the fps isn't exactly the same'''
        super()._cycle()
        t1 = time.perf_counter() - self.t0
        self.cycle_count = int(t1*self.fps)

    def move_effector(self, *args, **kwargs):
        current_pt = self.replay_task['cursor'][self.cycle_count]
        self.plant.set_endpoint_pos(current_pt)

    def _test_stop(self, ts):
        return super()._test_stop(ts) or self.cycle_count == len(self.replay_task)


class VideoPlayer(Window, Experiment):
    '''
    Plays video files saved in the visual_stimuli system. To add files, place
    them in the directory specified by the visual_stimuli system and they will be 
    automatically registered on startup. Requires ffmpeg and cv2.
    '''
    status = dict(
        wait=dict(start_video="play"),
        play=dict(stop=None, start_pause="pause"),
        pause=dict(stop=None, end_pause="play"),
    )
    state = "wait"

    media_file = traits.DataFile(object, desc="Visual stimulus video file. Add files to the directory specified " \
        "by the visual_stimuli system and they will be automatically registered on startup.", bmi3d_query_kwargs=dict(system__name='visual_stimuli'))
    audio_volume = traits.Float(1.0, desc="Playback volume from 0.0 to 1.0")
    start_time_sec = traits.Float(0.0, desc="Start playback from this time offset in seconds")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import cv2 # not a dependency for bmi3d, only needed for VideoPlayer

        self.event = None
        self._video_capture = None
        self._video_fps = 0.0
        self._next_frame_idx = 0
        self._video_finished = False
        self._audio_path = None
        self._audio_started = False
        self._playback_t0 = None
        self._video_start_frame = 0
        self._video_duration_sec = 0.0

        # Extract video metadata and prepare for playback
        # media_file can be a string (for testing/CLI) or a DataFile object
        if isinstance(self.media_file, str):
            media_file = os.path.expanduser(self.media_file)
        else:
            # It's a DataFile record; get the path
            media_file = self.media_file.get_path() if hasattr(self.media_file, 'get_path') else str(self.media_file)
        
        if not media_file:
            raise ValueError("media_file must be set to a valid local video path")
        if not os.path.isfile(media_file):
            raise IOError("media_file does not exist: %s" % media_file)

        self._video_capture = cv2.VideoCapture(media_file)
        if not self._video_capture.isOpened():
            raise RuntimeError("Could not open video file: %s" % media_file)
        self._video_fps = float(self._video_capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self._video_fps <= 0:
            self._video_fps = float(self.fps)

        total_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0 and self._video_fps > 0:
            self._video_duration_sec = total_frames / self._video_fps
        start_time_sec = max(0.0, self.start_time_sec)
        if self._video_duration_sec > 0:
            start_time_sec = min(start_time_sec, self._video_duration_sec)
        self._video_start_frame = max(0, int(start_time_sec * self._video_fps))

        # Extract audio
        self._audio_path = self._extract_audio_track(media_file, start_time_sec=start_time_sec)
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()

    def init(self):
        self.add_dtype('video_frame', 'u8', (1,))
        super().init()
        self.task_data['video_frame'] = -1

        # Keep native decode resolution for texture upload
        video_tex_w = int(self._video_capture.get(3))
        video_tex_h = int(self._video_capture.get(4))
        video_aspect = video_tex_w / video_tex_h if video_tex_h > 0 else 1.0
        if video_tex_w <= 0 or video_tex_h <= 0:
            video_tex_w, video_tex_h = 640, int(640 / max(video_aspect, 1e-6))
        self.video_size = (video_tex_w, video_tex_h)

        # Fit video plane inside screen while preserving aspect ratio (letterbox/pillarbox)
        screen_w = float(self.screen_cm[0])
        screen_h = float(self.screen_cm[1])
        screen_aspect = screen_w / screen_h
        if video_aspect >= screen_aspect:
            plane_w = screen_w
            plane_h = screen_w / video_aspect
        else:
            plane_h = screen_h
            plane_w = screen_h * video_aspect

        self.tex = Texture(
            np.zeros((self.video_size[1], self.video_size[0], 3), dtype=np.uint8),
            size=(self.video_size[0], self.video_size[1]),
            iformat=GL_RGB8,
            exformat=GL_RGB,
            dtype=GL_UNSIGNED_BYTE,
        )
        self.video_surface = TexPlane(plane_w, plane_h, tex=self.tex, specular_color=(0,0,0,0))
        self.video_surface.rotate_x(90).translate(-plane_w/2, 0, -plane_h/2)
        self.add_model(self.video_surface)

    def _extract_audio_track(self, media_file, start_time_sec=0.0):
        audio_fd, audio_path = tempfile.mkstemp(prefix="bmi3d_video_audio_", suffix=".wav")
        os.close(audio_fd)

        command = [
            "ffmpeg",
            "-y",
            "-ss", str(max(0.0, start_time_sec)),
            "-i", media_file,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            audio_path,
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            try:
                os.remove(audio_path)
            except OSError:
                pass
            print(f"No audio found in file {media_file}")
            print(result)
            return None
        return audio_path

    def _set_video_frame(self, frame):
        import cv2

        frame_rgb = np.flipud(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        self.tex.update(frame_rgb, size=(frame_rgb.shape[1], frame_rgb.shape[0]))

    def _end_wait(self):
        import cv2

        # Seek video decode to the requested starting point.
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, self._video_start_frame)
        self._next_frame_idx = self._video_start_frame
        self._video_finished = False

        # Start the audio playback
        if self._audio_path is not None:
            pygame.mixer.music.load(self._audio_path)
            pygame.mixer.music.set_volume(np.clip(self.audio_volume, 0.0, 1.0))
            pygame.mixer.music.play(loops=0)
            self._audio_started = True
        else:
            self._audio_started = False
            self._playback_t0 = self.get_time()

    def _test_start_video(self, ts):
        return True

    def _test_start_pause(self, ts):
        return self.pause

    def _test_end_pause(self, ts):
        return not self.pause

    def _start_pause(self):
        if self._audio_started:
            try:
                pygame.mixer.music.pause()
            except Exception:
                pass
        self.tex.update(np.zeros((3,3,3))) # blank the screen
        self.sync_event('PAUSE_START')

    def _end_pause(self):
        if self._audio_started:
            try:
                pygame.mixer.music.unpause()
            except Exception:
                pass
        else:
            self._playback_t0 += (self.get_time() - self.start_time) # omit the time spent in the pause state
        self.sync_event('PAUSE_END')

    def _test_stop(self, ts):
        return self._video_finished or super()._test_stop(ts)

    def _start_None(self):
        super()._start_None()

        # Stop audio playback
        if pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
            except Exception:
                pass

        # Release audio file
        if self._audio_path is not None and os.path.exists(self._audio_path):
            try:
                os.remove(self._audio_path)
            except OSError:
                pass
            self._audio_path = None

        # Release video capture
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

    def _target_frame_idx(self):

        if self._audio_started:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms < 0:
                if pygame.mixer.music.get_busy():
                    return max(0, self._next_frame_idx - 1)
                self._video_finished = True
                return max(0, self._next_frame_idx - 1)
            return self._video_start_frame + max(0, int((pos_ms / 1000.0) * self._video_fps))

        if self._playback_t0 is None:
            self._playback_t0 = self.get_time()
        elapsed = self.get_time() - self._playback_t0
        return self._video_start_frame + max(0, int(elapsed * self._video_fps))

    def _while_play(self):
        if self._video_finished:
            return

        target_idx = self._target_frame_idx()
        self.task_data['video_frame'] = target_idx
        self.reportstats['Video Frame'] = target_idx

        # Read and display video frames until we catch up to the target frame index
        while self._next_frame_idx <= target_idx and not self._video_finished:
            ok, frame = self._video_capture.read()
            if not ok:
                self._video_finished = True
                break
            self._set_video_frame(frame)
            self._next_frame_idx += 1

    @classmethod
    def get_desc(cls, params, report):
        runtime_sec = 0.0
        if isinstance(report, dict):
            runtime_sec = float(report.get('runtime', 0.0) or 0.0)
        elif isinstance(report, list) and len(report) > 0:
            runtime_sec = float(report[-1][-1] - report[0][-1])

        duration_min = runtime_sec / 60.0
        return "Video playback: {:.1f} min".format(duration_min)