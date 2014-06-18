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

class EndpointControlGoal(object):
    '''    Docstring    '''
    def __init__(self, ssm):
        '''    Docstring    '''
        assert ssm == train.endpt_2D_state_space
        self.ssm = ssm

    def __call__(self, target_pos, **kwargs):
        '''    Docstring    '''
        target_vel = np.array([0, 0, 0])
        offset_val = 1
        error = 0
        target_state = np.hstack([target_pos, target_vel, 1])
        return (target_state, error), True

    def reset(self):
        '''    Docstring    '''
        pass

class TwoLinkJointGoal(object):
    '''    Docstring    '''
    def __init__(self, ssm, shoulder_anchor, link_lengths):
        '''    Docstring    '''
        assert ssm == train.joint_2D_state_space
        self.ssm = ssm
        self.shoulder_anchor = shoulder_anchor
        self.link_lengths = link_lengths

    def __call__(self, target_pos, **kwargs):
        '''    Docstring    '''
        endpt_location = target_pos - self.shoulder_anchor
        joint_target_state = ik.inv_kin_2D(endpt_location, self.link_lengths[0], self.link_lengths[1])[0]

        target_state = []
        for state in self.ssm.state_names:
            if state == 'offset': # offset state is always 1
                target_state.append(1)
            elif re.match('.*?_v.*?', state): # Velocity states are always 0
                target_state.append(0) 
            elif state in joint_target_state.dtype.names:
                target_state.append(joint_target_state[state])
            else:
                raise ValueError('Unrecognized state: %s' % state)

        target_state = np.array(target_state)
        error = 0
        return (target_state, error), True

    def reset(self):
        '''    Docstring    '''
        pass        

class PlanarMultiLinkJointGoal(mp_calc.FuncProxy):
    '''    Docstring    '''
    def __init__(self, ssm, shoulder_anchor, kin_chain, multiproc=False, init_resp=None):
        '''    Docstring    '''
        def fn(target_pos, **kwargs):
            '''    Docstring    '''
            endpt_location = target_pos - shoulder_anchor
            joint_pos = kin_chain.inverse_kinematics(endpt_location, **kwargs)
            endpt_error = np.linalg.norm(kin_chain.endpoint_pos(joint_pos) - endpt_location)

            joint_pos *= -1 # the convention of the kin chain is different from that of the decoder/graphics..
            target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
            
            return target_state, endpt_error
        super(PlanarMultiLinkJointGoal, self).__init__(fn, multiproc=multiproc, waiting_resp='prev', init_resp=init_resp)


class PlanarMultiLinkJointGoalCached(mp_calc.FuncProxy):
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
                raise ValueError("Unknown target position!")

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
            raise ValueError("Unknown target position!")

        target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])
        
        # TODO These are wrong!
        endpt_error = 0 #np.linalg.norm(kin_chain.endpoint_pos(joint_pos) - endpt_location)

        return (target_state, endpt_error), True
