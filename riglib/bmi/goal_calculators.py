#!/usr/bin/python
'''
Calculate the goal state of the BMI for CLDA/assist/simulation
'''
import numpy as np
import train
from riglib import mp_calc

class EndpointControlGoal(object):
    def __init__(self, ssm):
        assert ssm == train.endpt_2D_state_space
        self.ssm = ssm

    def __call__(self, target_pos):
        target_vel = np.array([0, 0, 0])
        offset_val = 1
        return np.hstack([target_pos, target_vel, 1])


class TwoLinkJointGoal(object):
    def __init__(self, ssm, shoulder_anchor, link_lengths):
        assert ssm == train.joint_2D_state_space
        self.ssm = ssm
        self.shoulder_anchor = shoulder_anchor
        self.link_lengths = link_lengths

    def __call__(self, target_pos):
        endpt_location = target_pos - self.shoulder_anchor
        joint_target_state = ik.inv_kin_2D(endpt_location, self.link_lengths[0], self.link_lengths[1])[0]

        target_state = []
        for state in bmi_state_ls:
            if state == 'offset': # offset state is always 1
                target_state.append(1)
            elif re.match('.*?_v.*?', state): # Velocity states are always 0
                target_state.append(0) 
            elif state in joint_target_state.dtype.names:
                target_state.append(joint_target_state[state])
            else:
                raise ValueError('Unrecognized state: %s' % state)

        return np.array(target_state)


class PlanarMultiLinkJointGoal(mp_calc.FuncProxy):
    def __init__(self, ssm, shoulder_anchor, kin_chain, multiproc=False, init_resp=None):
        def fn(endpt_location, **kwargs):
            print endpt_location
            return kin_chain.inverse_kinematics(endpt_location - shoulder_anchor, **kwargs)
        super(PlanarMultiLinkJointGoal, self).__init__(fn, multiproc=multiproc, waiting_resp='prev', init_resp=init_resp)

