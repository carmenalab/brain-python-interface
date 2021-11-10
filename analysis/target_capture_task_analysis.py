#!/usr/bin/python
'''
Task-dependent performance measures (primarily for BMI tasks currently)
'''
import numpy as np
from scipy.stats import circmean, pearsonr
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import os
import tables

from riglib.bmi import robot_arms, train, kfdecoder, ppfdecoder
from db import dbfunctions
from db import dbfunctions as dbfn

from .performance_metrics import get_task_axis_error_measures


## Constants
min_per_sec = 1./60
seconds_per_min = 60
sec_per_min = 60
pi = np.pi


class Trials(object):
    def __init__(self, inds, length):
        self.inds = inds
        self.length = length

    def __iter__(self):
        return iter(self.inds)

    @property
    def bool(self):
        if not hasattr(self, '_full_inds'):
            _full_inds = np.zeros(self.length, dtype=bool)
            for st, end in self.inds:
                _full_inds[st:end] = 1
            self._full_inds = _full_inds
        return self._full_inds


class ManualControlMultiTaskEntry(dbfunctions.TaskEntry):
    '''
    Extension of dbfunctions TaskEntry class to calculate performance measures for the generic "target capture" task
    '''
    def __init__(self, *args, **kwargs):
        self.fixed = kwargs.pop('fixed', True)
        super(ManualControlMultiTaskEntry, self).__init__(*args, **kwargs)

        try:
            # Split the task messages into separate trials
            # A new trial starts in either the 'wait' state or when 'targ_transition' has a target_index of -1
            trial_start = np.logical_or(self.task_msgs['msg'] == b'wait', np.logical_and(self.task_msgs['msg'] == b'targ_transition', self.task_msgs['target_index'] == -1))
            trial_start_inds, = np.nonzero(trial_start)
            trial_end_inds = np.hstack([trial_start_inds[1:], len(trial_start)])
            self.trial_msgs = []
            for trial_st, trial_end in zip(trial_start_inds, trial_end_inds):
                self.trial_msgs.append(self.task_msgs[trial_st:trial_end])
            
            # Organize frame data
            frame_data = self.hdf.root.task[:]
            frame_data_dtype = np.dtype([('cursor', ('f8', 3)), ('manual_input', ('f8', 3))])
            if 'sync_square' in frame_data.dtype.names:
                frame_data_dtype = np.dtype([('cursor', ('f8', 3)), ('sync', '?')])
            frame_data_ext = np.zeros(len(frame_data), dtype=frame_data_dtype)
            for k in range(len(frame_data)):
                frame_data_ext[k]['cursor'] = frame_data[k]['cursor']
                frame_data_ext[k]['manual_input'] = frame_data[k]['manual_input']
                if 'sync_square' in frame_data.dtype.names:
                    frame_data_ext[k]['sync'] = frame_data[k]['sync_square']
            self.frame_data = frame_data_ext

        except:
            print("Couldn't process HDF file. Is it copied?")
            import traceback
            traceback.print_exc()

        if 'target_radius' in self.params:
            self.target_radius = self.params['target_radius']
        else:
            self.target_radius = 2.
        if 'cursor_radius' in self.params:
            self.cursor_radius = self.params['cursor_radius']
        else:
            self.cursor_radius = 0.4

        ### Update rate of task
        self.update_rate = 60.

    @property
    def reach_origin(self):
        return self.get_cached_attr('origin', self.calc_reach_origin)

    def calc_reach_origin(self):
        target = self.hdf.root.task[:]['target']
        origin = np.zeros_like(target)

        first_target_change = False
        prev_target = target[0]
        curr_origin = np.ones(3) * np.nan
        for t in range(len(target)):
            curr_target = target[t]
            if not np.array_equal(curr_target, prev_target):
                curr_origin = prev_target.copy()
                prev_target = curr_target

            origin[t] = curr_origin
        return origin

    @property
    def angular_error(self):
        return self.get_cached_attr('angular_error', self.calc_angular_error)

    def calc_angular_error(self):
        '''
        Compute the angular error between the cursor movement from one
        task loop iteration to the next (typically at 60 Hz). Angular error
        is with reference to the straight line between the cursor and the target
        '''
        # compute angles for each trial
        cursor = self.hdf.root.task[:]['cursor']
        target = self.hdf.root.task[:]['target']
        cursor_vel = np.diff(cursor, axis=0)
        int_dir = target - cursor

        dist_to_targ = np.array(map(np.linalg.norm, int_dir))
        window_angle = np.arctan2(self.target_radius, dist_to_targ)

        import geometry
        angles = geometry.angle(int_dir[:-1], cursor_vel, axis=0)
        angles = angles - window_angle[:-1]
        angles[angles < 0] = 0
        angles = np.hstack([angles, np.nan])
        return angles

    def get_targets(self):
        all_targets = self.hdf.root.task[:]['target']
        s = set(map(tuple, all_targets))
        return np.vstack(s)

    def plot_targets(self, ax=None, targets=None, facecolor='none', **kwargs):
        if ax == None:
            plt.figure()
            ax = plt.subplot(111)

        if targets == None:
            targets = self.get_targets()

        from pylab import Circle
        target_radius = self.target_radius

        patches = []
        for target in targets:
            c = Circle(target[[0,2]], radius=target_radius, facecolor=facecolor, **kwargs)
            patches.append(c)
            ax.add_patch(c)

        return ax, patches

    def get_fixed_decoder_task_msgs(self):
        try:
            return self._fixed_decoder_task_msgs, self._fixed_start
        except:
            hdf = self.hdf
            task_msgs = hdf.root.task_msgs[:]
            update_bmi_msgs = np.nonzero(task_msgs['msg'] == 'update_bmi')[0]
            if len(update_bmi_msgs) > 0:
                fixed_start = update_bmi_msgs[-1] + 1
            else:
                try:
                    assist_off = np.nonzero(hdf.root.task[:]['assist_level'] == 0)[0][0]
                except ValueError:
                    assist_off = 0
                except:
                    return np.zeros((0,), dtype=task_msgs.dtype), np.inf
                assist_off = filter(lambda k: task_msgs['time'][k] > assist_off, xrange(len(task_msgs)))[0]
                fixed_start = max(assist_off, 0)
            task_msgs = task_msgs[fixed_start:]
            self._fixed_decoder_task_msgs = task_msgs
            self._fixed_start = fixed_start
            return self._fixed_decoder_task_msgs, self._fixed_start
        #return task_msgs, fixed_start

    def get_plot_fnames(self):
        files = os.popen('ls /storage/plots | grep %s' % self.name)
        files = [f.rstrip() for f in files]
        return files

    def _from_hdf_get_trial_end_types(self, fixed=True):
        hdf = self.hdf
        target_index = hdf.root.task[:]['target_index']
        fixed = self.fixed
        if fixed:
            task_msgs, fixed_start = self.get_fixed_decoder_task_msgs()
        else:
            task_msgs = hdf.root.task_msgs[:]
            task_msgs = task_msgs[~(task_msgs['msg'] == 'update_bmi')]

        # Count the number of reward trials
        n_rewards = len(np.nonzero(task_msgs['msg'] == 'reward')[0])
        n_timeouts = len(np.nonzero(task_msgs['msg'] == 'timeout_penalty')[0])

        # TODO Number of trials
        hold_inds = np.nonzero(task_msgs['msg'] == 'hold')[0]
        target_seq_length = max(target_index) + 1
        hold_penalty_by_target = np.zeros(target_seq_length)
        for msg_ind in hold_inds:
            if task_msgs[msg_ind+1]['msg'] == 'hold_penalty':
                trial_targ_idx = target_index[task_msgs[msg_ind]['time']]
                hold_penalty_by_target[trial_targ_idx] += 1
        # Count the number of hold errors at each of the types of targets
        return dict(success=n_rewards, hold_error=hold_penalty_by_target, timeout=n_timeouts)

    def get_trial_end_types(self):
        # self.save()
        # hdf = tables.openFile('/storage/plots/fixed_bmi_performance.hdf', mode='r')
        # hdf.close()
        return self._from_hdf_get_trial_end_types()

    def get_rewards_per_min(self, window_size_mins=1.):
        '''
        Estimates rewards per minute. New estimates are made every 1./60 seconds
        using the # of rewards observed in the previous 'window_size_mins' minutes
        '''
        hdf = self.hdf
        task_msgs = hdf.root.task_msgs[:]
        reward_msgs = filter(lambda m: m[0] == 'reward', task_msgs)
        reward_on = np.zeros(hdf.root.task.shape)
        for reward_msg in reward_msgs:
            reward_on[reward_msg[1]] = 1

        # Hz
        window_size_updates = window_size_mins * seconds_per_min * self.update_rate
        conv = np.ones(window_size_updates) * 1./window_size_mins
        rewards_per_min = np.convolve(reward_on, conv, 'valid')
        tvec = np.arange(len(rewards_per_min)) * 1./self.update_rate + window_size_mins * seconds_per_min
        return tvec, rewards_per_min

    @property
    def clda_stop_time(self):
        try:
            task_msgs = self.hdf.root.task_msgs[:]
            last_update_msg_ind = np.nonzero(task_msgs['msg'] == 'update_bmi')[0][-1]
            last_update_msg = task_msgs[last_update_msg_ind]
            clda_stop = last_update_msg['time'] * 1./self.update_rate * min_per_sec
        except:
            clda_stop = 0
        return clda_stop

    @property
    def clda_stop_ind(self):
        task_msgs = self.hdf.root.task_msgs[:]
        last_update_msg_ind = np.nonzero(task_msgs['msg'] == 'update_bmi')[0][-1]
        last_update_msg = task_msgs[last_update_msg_ind]
        clda_stop = last_update_msg['time']
        return clda_stop

    def plot_rewards_per_min(self, ax=None, show=False, max_ylim=None, save=True, **kwargs):
        '''
        Make a plot of the rewards per minute
        '''
        import plotutil
        tvec, rewards_per_min = self.get_rewards_per_min(**kwargs)
        rewards_per_min = rewards_per_min[::900]
        tvec = tvec[::900]

        # find the time when CLDA turns off
        task_msgs = self.hdf.root.task_msgs[:]
        clda_stop = self.clda_stop_time

        if ax == None:
            plt.figure(figsize=(4,3))
            axes = plotutil.subplots(1, 1, return_flat=True, hold=True, left_offset=0.1)
            ax = axes[0]
        else:
            save = False

        try:
            # find the time when the assist turns off
            assist_level = self.hdf.root.task[:]['assist_level'].ravel()
            assist_stop = np.nonzero(assist_level == 0)[0][0]

            assist_stop *= min_per_sec * 1./self.update_rate # convert to min

            ax.axvline(assist_stop, label='Assist off', color='green', linewidth=2)
        except:
            pass
        ax.axvline(clda_stop, label='CLDA off', color='blue', linewidth=2, linestyle='--')
        ax.plot(tvec * min_per_sec, rewards_per_min, color='black', linewidth=2)
        if max_ylim == None:
            max_ylim = int(max(15, int(np.ceil(max(rewards_per_min)))))
        max_xlim = int(np.ceil(max(tvec * min_per_sec)))
        # plotutil.set_axlim(ax, [0, max_ylim], labels=range(max_ylim+1), axis='y')
        # plotutil.set_axlim(ax, [0, max_ylim], labels=range(0, max_ylim+1), axis='y')
        plotutil.set_xlim(ax, [0, max_xlim])
        plotutil.ylabel(ax, 'Rewards/min', offset=-0.08)
        plotutil.xlabel(ax, 'Time during block (min)')

        plotutil.legend(ax)
        ax.grid()

        if save: self.save_plot('rewards_per_min')

        if show:
            plt.show()

    @property
    def trials_per_min(self):
        return self.get_trial_end_types()['success']/self.length * sec_per_min

    @property
    def n_trials(self):
        return self.trial_end_types['success']

    @property
    def start_time(self):
        ''' Define the start tiem of the block. For a block with a fixed decoder,
        this is 0. For a block where the BMI changes, this is the time of the first fixed event
        '''
        task_msgs = self.hdf.root.task_msgs[:]
        if 'update_bmi' in task_msgs['msg']:
            task_msgs, _ = self.get_fixed_decoder_task_msgs()
            return task_msgs[0]['time'] * min_per_sec
        else:
            return 0.0

    @property
    def length(self):
        '''
        Length of session changes based on whether it was a 'fixed' block
        '''
        task_msgs, _ = self.get_fixed_decoder_task_msgs()
        rewardtimes = [r['time'] for r in task_msgs if r['msg']=='reward']
        if len(rewardtimes)>0:
            if self.fixed:
                return (rewardtimes[-1] * 1./self.update_rate - self.start_time)
            else:
                return rewardtimes[-1] * 1./self.update_rate
        else:
            return 0.0

    def label_trying(self, ds_factor=6):
        T = len(self.hdf.root.task) / ds_factor
        task_msgs = self.hdf.root.task_msgs[:]
        timeout_penalty_msg_inds = np.array(filter(lambda k: task_msgs[k]['msg'] == 'timeout_penalty', range(len(task_msgs))))
        #### exclude the last trial before a timeout
        labels = np.ones(T)
        for ind in timeout_penalty_msg_inds:
            # find the first 'hold' state before the timeout
            hold_ind = ind
            while not task_msgs[hold_ind]['msg'] == 'hold':
                hold_ind -= 1

            timeout_time = task_msgs[ind]['time'] / ds_factor
            hold_time = task_msgs[hold_ind]['time'] / ds_factor
            labels[hold_time : timeout_time] = 0
        ### Exclude the first 'target' state (return to center) after the timeout
        for ind in timeout_penalty_msg_inds:
            # find the first 'hold' state before the timeout
            hold_ind = ind
            while hold_ind < len(task_msgs) and not task_msgs[hold_ind]['msg'] == 'hold':
                hold_ind += 1

            if hold_ind < len(task_msgs):
                timeout_time = task_msgs[ind]['time'] / ds_factor
                hold_time = task_msgs[hold_ind]['time'] / ds_factor
                labels[timeout_time : hold_time] = 0
            else:
                labels[timeout_time:] = 0

        return labels.astype(bool)


class BMITaskEntry(ManualControlMultiTaskEntry):
    def __str__(self):
        return str(self.record) + '\nDecoder: %s' % (self.decoder.name)

    def __repr__(self):
        return self.__str__()

    def get_firing_rate_stats(self):
        mFR = np.mean(self.hdf.root.task[:]['spike_counts'], axis=0)
        sdFR = np.std(self.hdf.root.task[:]['spike_counts'], axis=0)
        return mFR, sdFR

    @property
    def assist_off_ind(self):
        assist_level = self.hdf.root.task[:]['assist_level'].ravel()
        try:
            assist_off_ind = np.nonzero(assist_level == 0)[0][0]
        except:
            # assist level never gets to 0
            assist_off_ind = np.nan
        return assist_off_ind

    def plot_loop_times(self, intended_update_rate=60.):
        loop_times = self.hdf.root.task[:]['loop_time'].ravel()
        plt.figure()
        axes = plotutil.subplots(1, 1, return_flat=True)
        plotutil.histogram_line(axes[0], loop_times, np.arange(0, 0.050, 0.0005))
        axes[0].axvline(1./intended_update_rate, color='black', linestyle='--')
        self.save_plot('loop_times')

    @property
    def perc_correct(self):
        trial_end_types = self.trial_end_types
        return float(trial_end_types['success']) / (trial_end_types['success'] + trial_end_types['timeout'] + sum(trial_end_types['hold_error'][1:]))

    def get_perc_correct(self, n_trials=None):
        if n_trials == None or n_trials == self.n_trials:
            return self.perc_correct
        else:
            # return the % correct within the first n_trials successful trials
            task_msgs, _ = self.get_fixed_decoder_task_msgs()

            n_rewards = 0
            n_timeouts = 0
            n_hold_errors = 0

            length = self.length
            target_index = self.hdf.root.task[:]['target_index']
            for msg in task_msgs:
                if n_rewards >= n_trials:

                    break
                elif msg['msg'] == 'reward':
                    n_rewards += 1
                elif msg['msg'] == 'timeout_penalty':
                    n_timeouts += 1
                elif msg['msg'] == 'hold_penalty':
                    trial_targ_idx = target_index[msg['time']-1]
                    if trial_targ_idx > 0: # ignore center hold errors
                        n_hold_errors += 1

            return float(n_rewards) / (n_rewards + n_timeouts + n_hold_errors)

    @property
    def decoder_type(self):
        from riglib.bmi import ppfdecoder, kfdecoder
        if isinstance(self.decoder, ppfdecoder.PPFDecoder):
            return 'PPF'
        elif isinstance(self.decoder, kfdecoder.KFDecoder):
            return 'KF'
        else:
            return 'unk'

    @property
    def training_tau(self):
        try:
            return dbfn.TaskEntry(self.decoder_record.entry).params['tau']
        except:
            return np.nan

    def cursor_speed(self, sl=slice(None)):
        cursor_pos = self.hdf.root.task[sl]['cursor']
        step_size = 1 if sl.step == None else sl.step
        cursor_vel = np.diff(cursor_pos, axis=0) * (self.update_rate/step_size)
        cursor_speed = np.array(map(np.linalg.norm, cursor_vel))
        return cursor_speed

    def get_ctrl_vecs(self):
        # get K
        if not hasattr(self, 'Ku'):
            F, K = self.decoder.filt.get_sskf()
            u = self.get_spike_counts()
            Ku = np.dot(K, u)
            self.Ku = Ku

        return self.Ku

    def get_decoder_state(self):
        if not hasattr(self, 'x_t'):
            if isinstance(self.decoder, kfdecoder.KFDecoder):
                self.x_t = np.mat(self.hdf.root.task[5::6]['decoder_state'][:,:,0].T)
            elif isinstance(self.decoder, ppfdecoder.PPFDecoder):
                try:
                    self.x_t = np.mat(np.hstack(self.hdf.root.task[:]['internal_decoder_state']))
                except:
                    self.x_t = np.mat(np.hstack(self.hdf.root.task[:]['decoder_state']))
            else:
                raise ValueError("decoder type?!?")
        return self.x_t

    def get_KF_active_BMI_motor_commands(self):
        '''
        KF Dynamics model: x_{t+1} = Ax_t + w_t
        KF update equation: x_{t+1|t+1} = Ax_{t|t} + K_t (y_{t+1} - CAx_{t|t})

        Therefore,
        w_{t+1|t+1} = K_t (y_{t+1} - CAx_{t|t})
                    = x_{t+1|t+1} - Ax_{t|t}
        (simultaneously estimate the newest motor command while refining the previous state estimate)
        '''
        if not hasattr(self, 'w'):
            y = np.mat(self.get_spike_counts())
            x = self.get_decoder_state()
            F, K = self.decoder.filt.get_sskf()
            C = np.mat(self.decoder.filt.C)
            A = np.mat(self.decoder.filt.A)
            self.w = y[:,1:] - C*A*x[:,:-1]

        return self.w

    def calc_Kyt(self):
        '''
        steady state kalman gain times obs
        '''
        y = np.mat(self.get_spike_counts())
        F, K = self.decoder.filt.get_sskf()
        K = np.mat(K)
        Kyt = K*y
        return Kyt

    @property
    def Kyt(self):
        return self.get_cached_attr('Kyt', self.calc_Kyt)

    def get_BMI_motor_commands(self):
        '''
        KF Dynamics model: x_{t+1} = Ax_t + w_t
        KF update equation: x_{t+1|t+1} = Ax_{t|t} + K_t (y_{t+1} - CAx_{t|t})

        Therefore,
        w_{t+1|t+1} = K_t (y_{t+1} - CAx_{t|t})
                    = x_{t+1|t+1} - Ax_{t|t}
        (simultaneously estimate the newest motor command while refining the previous state estimate)
        '''
        try:
            A = self.decoder.filt.A
        except:
            from db.tracker import models
            d = models.Decoder.objects.using(self.record._state.db).get(name=self.decoder_record.name.rstrip('_sskf'))
            A = d.load().filt.A

        if not hasattr(self, 'w_t'):
            x = self.get_decoder_state()
            A = np.mat(A)
            w_t = np.mat(np.zeros_like(x))
            w_t[:,:-1] = x[:,1:] - A*x[:,:-1]
            self.w_t = w_t

        return self.w_t

    def get_spike_counts(self, start=None, stop=None, binlen=None):
        if binlen == None:
            binlen = self.decoder.binlen

        if binlen > 1./self.update_rate: # Default bin lengths for graphics-driven tasks
            step = binlen/(1./self.update_rate)

        if not hasattr(self, 'u'):
            try:
                u_60hz = self.hdf.root.task[slice(None, None)]['spike_counts'][:,:,0]
                T = len(u_60hz)
                u = []
                for k in range(int(np.floor(T/step))):
                    u.append(np.sum(u_60hz[step*k: step*(k+1), :], axis=0))
                u = np.vstack(u).T
                self.u = u
            except:
                self.u = self.hdf.root.task[5::6]['lfp_power'][:,:,0].T

        return self.u


class CLDATaskEntry(BMITaskEntry):
    def __str__(self):
        try:
            decoder = self.get_decoders_trained_in_block() #dbfn.get_decoders_trained_in_block(self.record, dbname=self.dbname)
            if isinstance(decoder, list):
                decoder = decoder[0]
            return str(self.record) + '\nDecoder: %s' % decoder.name
        except:
            return super(CLDATaskEntry, self).__str__()

    def label_trying(self, *args, **kwargs):
        clda_stop_ind = self.clda_stop_ind / 6 #### #TODO REMOVE 60 Hz hardcoding!
        trying = super(CLDATaskEntry, self).label_trying(*args, **kwargs)
        trying[:clda_stop_ind] = 0
        return trying

    def gen_summary_plots(self):
        self.plot_rewards_per_min()

    def get_matching_state_transition_seq(self, seq):
        task_msgs = self.get_fixed_decoder_task_msgs()#  self.hdf.root.task_msgs[:]
        seq = np.array(seq, dtype='|S256')
        msg_list_inds = []
        trial_msgs = []
        epochs = []
        for k in range(len(task_msgs)-len(seq)):
            if np.all(task_msgs[k:k+len(seq)]['msg'] == seq):
                msg_list_inds.append(k)
                trial_msgs.append(task_msgs[k:k+len(seq)])
                epochs.append((task_msgs[k]['time'], task_msgs[k+len(seq)-1]['time']))
        return msg_list_inds, trial_msgs, epochs

    def plot_C_hist(self, param_fns=[lambda C_hist: C_hist[:,:,3], lambda C_hist: C_hist[:,:,5], lambda C_hist: C_hist[:,:,6], lambda C_hist: np.sqrt(C_hist[:, :, 3]**2 + C_hist[:,:,5]**2)],
            labels=['Change in x-vel tuning', 'Change in z-vel tuning', 'Change in baseline', 'Change in mod. depth']):
        '''
        Plot parameter trajectories for C
        '''
        C_hist = self.hdf.root.task[1:]['filt_C']
        n_units = C_hist.shape[1]
        n_blocks = int(np.ceil(float(n_units)/7))

        fig = plt.figure(facecolor='w', figsize=(8./3*len(param_fns), 2*n_blocks))
        axes = plotutil.subplots(n_blocks, len(param_fns), y=0.01)
        #, bottom_offset=0.01)
        #fig = plt.figure(figsize=(8, 2*n_units), facecolor='w')
        #axes = plotutil.subplots(n_units, len(param_fns), y=0.01) #, bottom_offset=0.01)

        for m, fn in enumerate(param_fns):
            for k in range(n_blocks):
                sl = slice(k*7, (k+1)*7, None)
                param_hist = fn(C_hist)[:,sl]
                param_hist_diff = param_hist - param_hist[0,:]
                axes[k,m].plot(param_hist_diff)
                axes[k,m].set_xticklabels([])
                if m == 0:
                    plotutil.ylabel(axes[k,m], 'Units %d-%d' % (sl.start, sl.stop-1))
                if k == n_blocks - 1:
                    plotutil.xlabel(axes[k,m], labels[m])

            lims = np.vstack(map(lambda ax: ax.get_ylim(), axes[:,m]))
            ylim = min(lims[:,0]), max(lims[:,1])
            plotutil.set_axlim(axes[:,m], ylim, axis='y')

        self.save_plot('clda_param_hist')

    def plot_C_hist_pds(self):
        C_hist_plot = self.hdf.root.task[1:10000:sec_per_min*self.update_rate]['filt_C']
        n_plots = C_hist_plot.shape[0]
        plt.figure(figsize=(3, 3*n_plots))
        axes = plotutil.subplots(n_plots, 1, return_flat=True, hold=True, aspect=1)
        for k in range(n_plots):
            self.decoder.plot_pds(C_hist_plot[k,:,:], ax=axes[k])
        self.save_plot('clda_param_hist_pds')

    def get_npz_param_hist(self, key, glue_fn=np.hstack):
        return np.array(glue_fn([x[key] for x in self.clda_param_hist]))

    @property
    def intended_kin(self):
        if not hasattr(self, '_intended_kin'):
            self._intended_kin = self.get_npz_param_hist('intended_kin', np.hstack)
        return self._intended_kin

    def intended_kin_norm(self, sl=slice(None, None)):
        return np.array(map(np.linalg.norm, self.intended_kin[sl, :].T))

    def cursor_speed(self, sl=None):
        if sl == None:
            sl = slice(None, None, self.update_rate)
        elif sl == 'assist_off':
            sl = slice(self.assist_off_ind, None, self.update_rate)
        return super(CLDATaskEntry, self).cursor_speed(sl)

    def plot_before_and_after_C(self):
        dec_before = self.decoder
        dec_after = dbfn.get_decoders_trained_in_block(self.id)
        plt.figure()
        axes = plotutil.subplots(1,2,return_flat=True, hold=True)
        dec_before.plot_C(ax=axes[0])
        dec_after.plot_C(ax=axes[1])

    @property
    def decoder(self):
        decoders = self.get_decoders_trained_in_block()
        if isinstance(decoders, list):
            return decoders[0]
        else:
            return decoders

    @property
    def seed_decoder(self):
        return dbfn.get_decoder(self.record)


## Calculate trials per min
def trials_per_min(task_entries):
    if not np.iterable(task_entries):
        task_entries = (task_entries,)

    length = 0
    n_rewards = 0

    for entry in task_entries:
        if isinstance(entry, int):
            te = _get_te(entry)
        else:
            te = entry

        n_rewards += te.n_rewards #te.get_trial_end_types()['success']
        length += float(len(te.hdf.root.task)) / te.update_rate
    return float(n_rewards)/length * seconds_per_min

def get_kf_blocks_after(id, **kwargs):
    blocks = dbfn.get_blocks_after(id, **kwargs)
    return filter(lambda x: _get_te(x).decoder_type == 'KF', blocks)

def get_ppf_blocks_after(id, **kwargs):
    blocks = dbfn.get_blocks_after(id, **kwargs)
    return filter(lambda x: _get_te(x).decoder_type == 'PPF', blocks)

def bits_per_sec(workspace_radius, target_radius):
    '''
    Calculate the difficulty of the BMI task in Fitts bits.
    This measure is defined in Gilja et al 2012, Nature neuroscience.
                Distance + Window
    bits = log2 -----------------
                     Window
    where 'Distance' is the distance between the center of the origin and
    center of the target. 'Window' is apparently slightly inaccurate in Gilja
    et al as the 'Window' in the numerator is the *radius* of the target and
    the 'Window' in the denominator is the *diameter* of the target
    '''
    workspace_radius = float(workspace_radius)
    return np.log2((workspace_radius + target_radius)/(2*target_radius))

def plot_targets(ax=None, targets=None, facecolor='none', radius=2, **kwargs):
    if ax == None:
        plt.figure()
        ax = plt.subplot(111)

    from pylab import Circle

    patches = []
    for target in targets:
        c = Circle(target[[0,2]], radius=radius, facecolor=facecolor, **kwargs)
        patches.append(c)
        ax.add_patch(c)

def sliding_average(data, window_size):
    window = np.ones(window_size)
    return 1./window_size * np.convolve(data, window, 'valid')[::window_size]

def reward_time_between_blocks(id0, id1):
    reward_time = 0
    for k in range(id0+1, id1):
        try:
            models.TaskEntry.objects.get(id=k)
            stuff = 1
        except:
            stuff = 0

        if stuff:
            te = _get_te(k)
            reward_time += te.total_reward_time
    return reward_time

def task_type(te):
    if te.decoder_type == 'KF':
        return 'KF'
    elif te.decoder_type == 'PPF':
        if not hasattr(te, 'feedback_rate'):
            return 'PPF'
        elif te.task_update_rate == 10:
            return 'LC'
        elif te.task_update_rate == 60:
            return 'LF'


def _get_te(te, **kwargs):
    # dbname = kwargs.pop('dbname', 'default')
    te = dbfn.TaskEntry(te, **kwargs)
    try:
        return tasks[te.record.task.name](te.record.id, **kwargs)
    except:
        return te

def summarize_bmi_performance(date, **kwargs):
    ''' For a given date, print out a summary of the BMI performance
    '''
    for block in dbfn.get_bmi_blocks(date, **kwargs):
        te = _get_te(block)
        print(te)
        print(te.summary())

def summarize_performance(blocks, **kwargs):
    ''' For a given date, print out a summary of the BMI performance
    '''
    for block in blocks:
        te = _get_te(block)
        print(te)
        print(te.summary())

def compare_perc_correct(te1, te2):
    from scipy import stats
    end_types1 = te1.get_trial_end_types()
    end_types2 = te2.get_trial_end_types()
    n_hold_errors = np.sum(end_types['hold_error'][1:]) # No. of hold errors, excluding the first target in the trial target sequence
    def fn(end_types): return (end_types['success'], n_hold_errors)
    print(fn(end_types1))
    return stats.chi2_contingency(np.array([fn(end_types1), fn(end_types2)]))


def dir_change(hdf, step=6):
    boundaries = dbfunctions.get_center_out_reach_inds(hdf)

    n_trials = boundaries.shape[0]
    vel_angle_diff = [None] * n_trials
    cursor = hdf.root.task[:]['cursor']
    for k, (st, end) in enumerate(boundaries):
        cursor_pos_tr = cursor[st:end:step, [0,2]]
        vel = np.diff(cursor_pos_tr, axis=0)
        vel_angle = np.arctan2(vel[:,1], vel[:,0])
        vel_angle_diff[k] = np.diff(vel_angle)

    vel_angle_diff_concat = np.hstack(vel_angle_diff)
    mean = circmean(np.abs(vel_angle_diff_concat), high=2*np.pi, low=-2*np.pi)
    print(mean)
    return vel_angle_diff, mean

def edge_detect(vec, edge_type='pos'):
    """ Edge detector for a 1D array

    Example:

    vec = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...]
                       ^           ^
                       ^           ^
                      pos         neg
                      edge        edge

    vec         : 1D array
    edge_type   : {'pos', 'neg'}
    """
    if np.ndim(vec) > 1:
        vec = vec.reshape(-1)
    T = len(vec)
    edges = np.zeros(T)
    for t in range(1,T):
        if edge_type == 'pos':
            if vec[t] and not vec[t-1]:
                edges[t] = 1
        elif edge_type == 'neg':
            if vec[t-1] and not vec[t]:
                edges[t] = 1
    return edges

def _count_switches(vec):
    """ vec is an array of binary variables (0,1). The number of switches
    between 1's and 0's is counted
    """
    return len(np.nonzero(edge_detect(vec, 'pos'))[0]) + len(np.nonzero(edge_detect(vec, 'neg'))[0])


def get_trial_end_types(entry):
    entry = lookup_task_entries(entry)
    hdf = get_hdf(entry)
    task_msgs = get_fixed_decoder_task_msgs(hdf)

    # number of successful trials
    reward_msgs = filter(lambda m: m[0] == 'reward', task_msgs)
    n_success_trials = len(reward_msgs)

    # number of hold errors
    hold_penalty_inds = np.array(filter(lambda k: task_msgs[k][0] == 'hold_penalty', range(len(task_msgs))))
    msg_before_hold_penalty = task_msgs[(hold_penalty_inds - 1).tolist()]
    n_terminus_hold_errors = len(filter(lambda m: m['msg'] == 'terminus_hold', msg_before_hold_penalty))
    n_origin_hold_errors = len(filter(lambda m: m['msg'] == 'origin_hold', msg_before_hold_penalty))

    # number of timeout trials
    timeout_msgs = filter(lambda m: m[0] == 'timeout_penalty', task_msgs)
    n_timeout_trials = len(timeout_msgs)

    return n_success_trials, n_terminus_hold_errors, n_timeout_trials, n_origin_hold_errors

def get_hold_error_rate(task_entry):
    hold_error_rate = float(n_terminus_hold_errors) / n_success_trials
    return hold_error_rate

def get_fixed_decoder_task_msgs(hdf):
    task_msgs = hdf.root.task_msgs[:]
    update_bmi_msgs = np.nonzero(task_msgs['msg'] == 'update_bmi')[0]
    if len(update_bmi_msgs) > 0:
        fixed_start = update_bmi_msgs[-1] + 1
    else:
        fixed_start = 0
    task_msgs = task_msgs[fixed_start:]
    return task_msgs

def get_center_out_reach_inds(hdf, fixed=True):
    if fixed:
        task_msgs = get_fixed_decoder_task_msgs(hdf)
    else:
        task_msgs = hdf.root.task_msgs[:]

    n_msgs = len(task_msgs)
    terminus_hold_msg_inds = np.array(filter(lambda k: task_msgs[k]['msg'] == 'terminus_hold', range(n_msgs)))
    if terminus_hold_msg_inds[0] == 0: # HACK mid-trial start due to CLDA
        terminus_hold_msg_inds = terminus_hold_msg_inds[1:]
    terminus_msg_inds = terminus_hold_msg_inds - 1

    boundaries = np.vstack([task_msgs[terminus_msg_inds]['time'],
                            task_msgs[terminus_hold_msg_inds]['time']]).T
    return boundaries

def get_movement_durations(task_entry):
    '''
    Get the movement durations of each trial which enters the 'terminus_hold'
    state
    '''
    hdf = get_hdf(task_entry)
    boundaries = get_center_out_reach_inds(hdf)

    return np.diff(boundaries, axis=1) * self.update_rate

def get_movement_error(task_entry):
    '''
    Get movement error
    '''
    task_entry = lookup_task_entries(task_entry)
    reach_trajectories = get_reach_trajectories(task_entry)

    n_trials = len(reach_trajectories)

    ME = np.array([np.mean(np.abs(x[1, ::6])) for x in reach_trajectories])
    MV = np.array([np.std(np.abs(x[1, ::6])) for x in reach_trajectories])

    return ME, MV

def get_total_movement_error(task_entry):
    task_entry = lookup_task_entries(task_entry)
    reach_trajectories = get_reach_trajectories(task_entry)
    total_ME = np.array([np.sum(np.abs(x[1, ::6])) for x in reach_trajectories])
    return total_ME


def edge_detect(vec, edge_type='pos'):
    """ Edge detector for a 1D array

    Example:

    vec = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...]
                       ^           ^
                       ^           ^
                      pos         neg
                      edge        edge

    vec         : 1D array
    edge_type   : {'pos', 'neg'}
    """
    if np.ndim(vec) > 1:
        vec = vec.reshape(-1)
    T = len(vec)
    edges = np.zeros(T)
    for t in range(1,T):
        if edge_type == 'pos':
            if vec[t] and not vec[t-1]:
                edges[t] = 1
        elif edge_type == 'neg':
            if vec[t-1] and not vec[t]:
                edges[t] = 1
    return edges

def _count_switches(vec):
    """ vec is an array of binary variables (0,1). The number of switches
    between 1's and 0's is counted
    """
    return len(np.nonzero(edge_detect(vec, 'pos'))[0]) + len(np.nonzero(edge_detect(vec, 'neg'))[0])

def get_direction_change_counts(entry):
    entry = lookup_task_entries(entry)
    reach_trajectories = get_reach_trajectories(entry)

    n_trials = len(reach_trajectories)

    ODCs = np.array([_count_switches( 0.5*(np.sign(np.diff(x[0,::6])) + 1) ) for x in reach_trajectories])
    MDCs = np.array([_count_switches( 0.5*(np.sign(np.diff(x[1,::6])) + 1) ) for x in reach_trajectories])

    return MDCs, ODCs


def plot_trajectories(task_entry, ax=None, show=False, **kwargs):
    hdf = get_hdf(task_entry)
    boundaries = get_center_out_reach_inds(hdf)
    targets = hdf.root.task[:]['target']
    cursor = hdf.root.task[:]['cursor']

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    n_trials = boundaries.shape[0]
    for k, (st, end) in enumerate(boundaries):
        trial_target = targets[st][[0,2]]
        angle = -np.arctan2(trial_target[1], trial_target[0])

        # counter-rotate trajectory
        cursor_pos_tr = cursor[st:end, [0,2]]
        trial_len = cursor_pos_tr.shape[0]
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        cursor_pos_tr_rot = np.vstack([np.dot(R, cursor_pos_tr[k,:]) for k in range(trial_len)])
        ax.plot(cursor_pos_tr_rot[:,0], cursor_pos_tr_rot[:,1], **kwargs)

    if show:
        plt.show()

def get_workspace_size(task_entry):
    '''
    Get movement error
    '''
    hdf = get_hdf(task_entry)
    targets = hdf.root.task[:]['target']
    print(targets.min(axis=0))
    print(targets.max(axis=0))

def plot_dist_to_targ(task_entry, reach_trajectories=None, targ_dist=10., plot_all=False, ax=None, target=None, update_rate=60., decoder_rate=10., **kwargs):
    task_entry = dbfn.lookup_task_entries(task_entry)
    if reach_trajectories == None:
        reach_trajectories = task_entry.get_reach_trajectories()
    if target == None:
        target = np.array([targ_dist, 0])
    trajectories_dist_to_targ = [map(np.linalg.norm, traj.T - target) for traj in reach_trajectories]

    step = update_rate/decoder_rate
    trajectories_dist_to_targ = map(lambda x: x[::step], trajectories_dist_to_targ)
    max_len = np.max([len(traj) for traj in trajectories_dist_to_targ])
    n_trials = len(trajectories_dist_to_targ)

    # TODO use masked arrays
    data = np.ones([n_trials, max_len]) * np.nan
    for k, traj in enumerate(trajectories_dist_to_targ):
        data[k, :len(traj)] = traj

    from scipy.stats import nanmean, nanstd
    mean_dist_to_targ = np.array([nanmean(data[:,k]) for k in range(max_len)])
    std_dist_to_targ = np.array([nanstd(data[:,k]) for k in range(max_len)])

    if ax == None:
        plt.figure()
        ax = plt.subplot(111)

    # time vector, assuming original screen update rate of 60 Hz
    time = np.arange(max_len)*0.1
    if plot_all:
        for dist_to_targ in trajectories_dist_to_targ:
            ax.plot(dist_to_targ, **kwargs)
    else:
        ax.plot(time, mean_dist_to_targ, **kwargs)

    import plotutil
    #plotutil.set_ylim(ax, [0, targ_dist])
    plotutil.ylabel(ax, 'Distance to target')
    plotutil.xlabel(ax, 'Time (s)')
    plt.draw()
