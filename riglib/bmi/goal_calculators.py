#!/usr/bin/python
'''
Calculate the goal state of the BMI for CLDA/assist/simulation
'''
import numpy as np
import train

class BMIGoalState(object):
    def __init__(self, ssm):
        self.ssm = ssm


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
        assert ssm = train.joint_2D_state_space
        self.ssm = ssm
        self.shoulder_anchor = shoulder_anchor
        self.link_lengths = link_lengths

    def __call__(self, target_pos):
        endpt_location = target_pos - self.shoulder_anchor
        joint_target_state = ik.inv_kin_2D(endpt_location, self.link_lengths[0], self.link_lengths[1])[0]




    def get_target_BMI_state(self, bmi_state_ls):
        '''
        For CLDA purposes, this method allows the task to define the target
        'state' of the BMI. Note that this is different from the *intended*
        state
        '''
        # TODO extend this to 3D!
        endpt_target_state = self.target_location
        dtype=[('hand_px', np.float64), ('hand_py', np.float64), ('hand_pz', np.float64)]
        endpt_target_state = np.zeros((1,), dtype)
        endpt_target_state[0] = self.target_location

        endpt_location = self.target_location - self.shoulder_anchor
        joint_target_state = ik.inv_kin_2D(endpt_location, self.arm_link_lengths[1], self.arm_link_lengths[0])[0]

        target_state = []
        for state in bmi_state_ls:
            if state == 'offset': # offset state is always 1
                target_state.append(1)
            elif re.match('.*?_v.*?', state): # Velocity states are always 0
                target_state.append(0) 
            elif state in endpt_target_state.dtype.names:
                target_state.append(endpt_target_state[0][state])
            elif state in joint_target_state.dtype.names:
                target_state.append(joint_target_state[state])
            else:
                raise ValueError('Unrecognized state: %s' % state)

        if not len(target_state) == len(bmi_state_ls):
            raise ValueError("A state got lost somehow....")

        return np.array(target_state).reshape(-1,1)