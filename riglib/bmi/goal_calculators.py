#!/usr/bin/python
'''
Classes to determine the "goal" during a BMI task. Knowledge of the
task goal is required for many versions of assistive control (see assist.py) and
many versions of CLDA (see clda.py). 
'''
import numpy as np
import train
from riglib import mp_calc
from riglib.stereo_opengl import ik
import re
import pickle
from riglib.bmi import state_space_models

# TODO -- does ssm really need to be passed as an argument into __init__?
# maybe just make it an optional kwarg for the classes that really need it

class GoalCalculator(object):
    def reset(self):
        pass    

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
        target_vel = np.zeros_like(target_pos)
        offset_val = 1
        error = 0
        target_state = np.hstack([target_pos, target_vel, 1]).reshape(-1, 1)
        return target_state

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
            print endpt_error

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
