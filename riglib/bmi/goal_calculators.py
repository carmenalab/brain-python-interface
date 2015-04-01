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
    def __init__(self, ssm):
        '''
        Constructor for ZeroVelocityGoal
    
        Parameters
        ----------
        ssm : state_space_models.StateSpace instance
            The state-space model of the Decoder that is being assisted/adapted. Currently unused
    
        Returns
        -------
        ZeroVelocityGoal instance
        '''
        # assert ssm == train.endpt_2D_state_space
        self.ssm = ssm

    def __call__(self, target_pos, **kwargs):
        '''
        Docstring
    
        Parameters
        ----------
        target_pos : DATA_TYPE
            ARG_DESCR
        kwargs : optional kwargs
            ARG_DESCR
    
        Returns
        -------
        
        '''
        target_vel = np.zeros_like(target_pos)
        offset_val = 1
        error = 0
        target_state = np.hstack([target_pos, target_vel, 1])
        return (target_state, error), True

class ZeroVelocityAccelGoal(ZeroVelocityGoal):
    def __call__(self, target_pos, **kwargs):
        target_vel = np.zeros_like(target_pos)
        target_acc = np.zeros_like(target_pos)
        offset_val = 1
        error = 0
        target_state = np.hstack([target_pos, target_vel, target_acc, 1])
        return (target_state, error), True        


class PlanarMultiLinkJointGoal(GoalCalculator, mp_calc.FuncProxy):
    '''    Docstring    '''
    def __init__(self, ssm, shoulder_anchor, kin_chain, multiproc=False, init_resp=None):
        '''    Docstring    '''
        def fn(target_pos, **kwargs):
            '''    Docstring    '''
            joint_pos = kin_chain.inverse_kinematics(target_pos, **kwargs)
            endpt_error = np.linalg.norm(kin_chain.endpoint_pos(joint_pos) - target_pos)

            target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
            
            return target_state, endpt_error
        super(PlanarMultiLinkJointGoal, self).__init__(fn, multiproc=multiproc, waiting_resp='prev', init_resp=init_resp)

class PlanarMultiLinkJointGoalCached(GoalCalculator, mp_calc.FuncProxy):
    '''    Docstring    '''
    def __init__(self, ssm, shoulder_anchor, kin_chain, multiproc=False, init_resp=None, **kwargs):
        '''    Docstring    '''
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

            if joint_pos == None:
                raise ValueError("Unknown target position!: %s" % str(target_pos))

            target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
            
            int_endpt_location = target_pos - shoulder_anchor
            endpt_error = np.linalg.norm(kin_chain.endpoint_pos(-joint_pos) - int_endpt_location)
            print endpt_error

            return (target_state, endpt_error)

        super(PlanarMultiLinkJointGoalCached, self).__init__(fn, multiproc=multiproc, waiting_resp='prev', init_resp=init_resp)

    def __call__(self, target_pos, **kwargs):
        '''    Docstring    '''
        joint_pos = None
        for pos in self.cached_data:
            if np.linalg.norm(target_pos - np.array(pos)) < 0.001:
                joint_pos = self.cached_data[pos]
                # possible_joint_pos = self.cached_data[pos]
                # ind = np.random.randint(0, len(possible_joint_pos))
                # joint_pos = possible_joint_pos[ind]
                # break

        # print joint_pos

        if joint_pos == None:
            raise ValueError("Unknown target position!: %s" % str(target_pos))

        target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
        
        # TODO These are wrong!
        endpt_error = 0 #np.linalg.norm(kin_chain.endpoint_pos(joint_pos) - endpt_location)

        return (target_state, endpt_error), True
