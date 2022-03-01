#!/usr/bin/python
'''
Classes to determine the "goal" during a BMI task. Knowledge of the
task goal is required for many versions of assistive control (see assist.py) and
many versions of CLDA (see clda.py). 
'''
import numpy as np
from .. import mp_calc
import pickle

# TODO -- does ssm really need to be passed as an argument into __init__?
# maybe just make it an optional kwarg for the classes that really need it

class GoalCalculator(object):
    def reset(self):
        pass    

class Obs_Goal_Calc(GoalCalculator):
    def __init__(self, ssm=None, **kwargs):
        self.ssm = ssm
        import os
        self.pre_obs = True
        self.mid_speed = kwargs.pop('mid_targ_speed', 10)
        self.mid_targ_rad = kwargs.pop('mid_targ_rad', 6)
        self.targ_cnt = 0
        self.pre_obs_targ_state = None
        self.post_obs_targ_state = None

    def clear(self):
        self.pre_obs_targ_state = None
        self.post_obs_targ_state = None    
        print('CLEAR')    

    def __call__(self, target_pos, **kwargs):
        #Use q_start th
        pos = kwargs.pop('q_start')

        #if past obstacle midline: 
        
        if 'center_pos' in kwargs:
            obstacle_center = kwargs['center_pos'] + (target_pos - kwargs['center_pos'])*.5
            center = kwargs['center_pos']
        else:
            obstacle_center = target_pos/2.
            center = np.zeros((3, ))

        target_pos = target_pos.round(1)
        try:
            slope = -1*1./((target_pos[2] - center[2])/(target_pos[0]-center[0]))
        except:
            slope = np.inf
            
        #if ((np.abs(slope) != np.inf) and (np.abs(slope) != np.nan) and np.abs(slope)!=0):
        pre_obs = self.fcn_det(slope, obstacle_center, pos, center, target_pos)
            #print 'pre_obs: ', pre_obs, slope
        # else:
        #     #print 'division by zero!'
        #     if target_pos[0] ==0:
        #         #Division by zero
        #         pre_obs = False
        #         if np.abs(pos[2]) < (np.abs(obstacle_center[2])-.2): pre_obs = True
        #     elif target_pos[2] == 0:
        #         pre_obs = False
        #         if np.abs(pos[0]) < (np.abs(obstacle_center[0]) -.2): pre_obs = True
        #     else: 
        #         Exception('Not vertical or horiz. line causing divide by zero --> error')

        if pre_obs:
            if 1:
            #if self.pre_obs_targ_state is None:
                obs_ang = np.angle(obstacle_center[0]-center[0] + 1j*(obstacle_center[2]-center[2]))
                obs_r = np.abs((obstacle_center[0]-center[0]) + 1j*(obstacle_center[2]-center[2]))
                
                if self.ccw_fcn(pos, obstacle_center): 
                    targ_vect_ang = np.pi/2
                else:
                    targ_vect_ang = -1*np.pi/2

                target_state_pos = obstacle_center + self.mid_targ_rad*(np.array([np.cos(targ_vect_ang+obs_ang), 0, np.sin(targ_vect_ang+obs_ang)]))
                target_vel = self.mid_speed*np.array([np.cos(obs_ang), 0, np.sin(obs_ang)])
                target_state = np.hstack((target_state_pos, target_vel, 1)).reshape(-1, 1)
                self.pre_obs_targ_state = target_state
            else:
                target_state = self.pre_obs_targ_state

        else:
            if 1:
            #if self.post_obs_targ_state is None:
                target_vel = np.zeros_like(target_pos)
                offset_val = 1
                target_state = np.hstack([target_pos, target_vel, 1]).reshape(-1, 1)
                
                if self.pre_obs_targ_state is not None:
                    self.post_obs_targ_state = target_state
            else:
                target_state = self.post_obs_targ_state

        error = 0

        # if self.pre_obs != pre_obs:
        #     self.pre_obs = pre_obs
        #     print self.pre_obs, target_state

        return (target_state, error), True


    def fcn_det(self, slope, pt_on_line, test_pt, center, target):
        d_center = np.sqrt(np.sum((test_pt - center)**2))
        d_target = np.sqrt(np.sum((test_pt - target)**2))
        if d_center < d_target:
            return True
        else:
            return False


        # abs_pt = np.abs(pt_on_line)
        # abs_test = np.abs(test_pt)
        # slope = -1*np.abs(slope)

        # b = abs_pt[2] - slope*abs_pt[0]
       
        # if abs_test[2] +0.3 < (b + slope*abs_test[0]):
        #     return True
        # else:
        #     return False

        # zz = False
        # if 0 < b: zz = True

        # if test_pt[2] < 0:
        #     eps = -.2
        # else:
        #     eps = .2

        # test = False
        # if (test_pt[2]+eps) < ((slope*test_pt[0]) + b): test = True

        # if zz!=test:
        #     return True
        # else:
        #     return False

    def ccw_fcn(self, pos_test, pos_ref):
        theta1 = np.angle(pos_test[0] + 1j * pos_test[2])
        theta2 = np.angle(pos_ref[0]+ 1j*pos_ref[2])

        if pos_ref[0] < 0 and pos_test[0] < 0:
            if pos_ref[2] < 0 and pos_test[2] >0:
                theta2 += 2*np.pi
            elif pos_ref[2] > 0 and pos_test[2] < 0:
                theta1 += 2*np.pi
        return theta1 > theta2

class ZeroVelocityGoal(GoalCalculator):
    '''
    Assumes that the target state of the BMI is to move to the task-specified position with zero velocity
    '''
    def __init__(self, ssm=None):
        '''
        Constructor for ZeroVelocityGoal
    
        Parameters
        ----------
        ssm : state_space_models.StateSpace instance
            The state-space model of the Decoder that is being assisted/adapted. Not needed for this particular method
    
        Returns
        -------
        ZeroVelocityGoal instance
        '''
        try:
            self.ssm = ssm()
        except:
            self.ssm = ssm

    def __call__(self, target_pos, **kwargs):
        '''
        Calculate the goal state [p, 0, 1] where p is the n-dim position and 0 is the n-dim velocity
    
        Parameters
        ----------
        target_pos : np.ndarray
            Optimal position, in generalized coordinates (i.e., need not be cartesian coordinates)
        kwargs : optional kwargs
            These are ignored, just present for function call compatibility
    
        Returns
        -------
        np.ndarray
            (N, 1) indicating the target state
        '''

        # Add zero velocity if needed: 
        n_pos_vel_states = int(self.ssm.n_states) - 1
        if len(target_pos) < n_pos_vel_states :
            target_vel = np.zeros_like(target_pos)
            offset_val = 1
            target_state = np.hstack([target_pos, target_vel, 1]).reshape(-1, 1)
        elif len(target_pos) == n_pos_vel_states:
            target_state = np.hstack([target_pos, 1]).reshape(-1, 1)
        else:
            target_state = np.hstack(target_pos).reshape(-1, 1) # don't add offset
        error = 0
        return (target_state, error), True

class ZeroVelocityGoal_ismore(GoalCalculator):
    def __init__(self, ssm=None, pause_states=[]):
        try:
            self.ssm = ssm()
        except:
            self.ssm = ssm

        self.pause_states = pause_states

    def __call__(self, target_pos, state, **kwargs):
        if state in self.pause_states:
            target_state = kwargs['current_state']
            error = 0
        else:
            n_pos_vel_states = int(self.ssm.n_states) - 1
            if len(target_pos) < n_pos_vel_states :
                target_vel = np.zeros_like(target_pos)
                offset_val = 1
                target_state = np.hstack([target_pos, target_vel, 1]).reshape(-1, 1)
            else:
                target_state = np.hstack([target_pos, 1]).reshape(-1, 1)
            error = 0
        return (target_state, error), True            

class ZeroVelocityAccelGoal(ZeroVelocityGoal):
    '''
    Similar to ZeroVelocityGoal, but used for a second order system where you also want the goal acceleration to be zero.
    '''
    def __call__(self, target_pos, **kwargs):
        '''
        See ZeroVelocityGoal.__call__ for argument documentation
        '''
        target_vel = np.zeros_like(target_pos)
        target_acc = np.zeros_like(target_pos)
        offset_val = 1
        error = 0
        target_state = np.hstack([target_pos, target_vel, target_acc, 1])
        return (target_state, error), True        


class PlanarMultiLinkJointGoal(GoalCalculator, mp_calc.FuncProxy):
    '''
    Looks up goal configuration for a redundant system based on the endpoint goal and tries to find the closest solution.

    DEPRECATED: The method implemented has not been used for a long time, and is not the best method for achieving finding the "closest" config space solution as desired.
    '''
    def __init__(self, ssm, shoulder_anchor, kin_chain, multiproc=False, init_resp=None):
        def fn(target_pos, **kwargs):
            joint_pos = kin_chain.inverse_kinematics(target_pos, **kwargs)
            endpt_error = np.linalg.norm(kin_chain.endpoint_pos(joint_pos) - target_pos)

            target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
            
            return target_state, endpt_error
        super(PlanarMultiLinkJointGoal, self).__init__(fn, multiproc=multiproc, waiting_resp='prev', init_resp=init_resp)

class PlanarMultiLinkJointGoalCached(GoalCalculator, mp_calc.FuncProxy):
    '''
    Determine the goal state of a redundant system by look-up-table, i.e. redundancy is collapsed 
    by arbitrary mapping between redudnant target space and configuration space

    TODO: since multiprocessing is not required for this class, it needs to do a better job of hiding the multiprocessing.
    '''
    def __init__(self, ssm, shoulder_anchor, kin_chain, multiproc=False, init_resp=None, **kwargs):
        '''
        Constructor for PlanarMultiLinkJointGoalCached

        Parameters
        ----------
        ssm : state_space_models.StateSpace instance
        shoulder_anchor : np.ndarray of shape (3,)
            Position of the manipulator anchor
        kin_chain : robot_arms.KinematicChain instance
            Object representing the kinematic chain linkages (D-H parameters)
        multiproc : bool, optional, default=False
            Should leave this false for this 'cached' method
        init_resp : None
            Ignore this if multiproc=False, as directed above.
        kwargs : optional keyword arguments
            Can pass in 'goal_cache_block' to specify from which task entry to grab the cache file. WARNING: this is currently commented out

        Returns
        -------
        PlanarMultiLinkJointGoalCached instance

        '''
        self.ssm = ssm
        self.shoulder_anchor = shoulder_anchor
        self.kin_chain = kin_chain
        if 0: #'goal_cache_block' in kwargs:
            goal_cache_block = kwargs.pop('goal_cache_block')
            self.cached_data = pickle.load(open('/storage/assist_params/tentacle_cache_%d.pkl' % goal_cache_block))
        else:
            self.cached_data = pickle.load(open('/storage/assist_params/tentacle_cache3.pkl'))

        def fn(target_pos, **kwargs):
            '''    Docstring    '''
            joint_pos = None
            for pos in self.cached_data:
                if np.linalg.norm(target_pos - np.array(pos)) < 0.001:
                    possible_joint_pos = self.cached_data[pos]
                    ind = np.random.randint(0, len(possible_joint_pos))
                    joint_pos = possible_joint_pos[ind]
                    break

            if joint_pos is None:
                raise ValueError("Unknown target position!: %s" % str(target_pos))

            target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
            
            int_endpt_location = target_pos - shoulder_anchor
            endpt_error = np.linalg.norm(kin_chain.endpoint_pos(-joint_pos) - int_endpt_location)
            print(endpt_error)

            return (target_state, endpt_error)

        super(PlanarMultiLinkJointGoalCached, self).__init__(fn, multiproc=multiproc, waiting_resp='prev', init_resp=init_resp)

    def __call__(self, target_pos, **kwargs):
        '''
        Calculate the goal state [p, 0, 1] where p is the n-dim position and 0 is the n-dim velocity
        p is the configuration space position, which must be looked up based on the target_pos 
        (not a one-to-one mapping in general)
    
        Parameters
        ----------
        target_pos : np.ndarray
            Optimal position, in generalized coordinates (i.e., need not be cartesian coordinates)
        kwargs : optional kwargs
            These are ignored, just present for function call compatibility
    
        Returns
        -------
        np.ndarray
            (N, 1) indicating the target state
        '''
        joint_pos = None
        for pos in self.cached_data:
            if np.linalg.norm(target_pos - np.array(pos)) < 0.001:
                joint_pos = self.cached_data[pos]

        if joint_pos == None:
            raise ValueError("Unknown target position!: %s" % str(target_pos))

        target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
        
        endpt_error = 0

        return (target_state, endpt_error), True
