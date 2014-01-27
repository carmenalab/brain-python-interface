#!/usr/bin/python
'''
Calculate the goal state of the BMI for CLDA/assist/simulation
'''
import numpy as np
import train

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


# create process 
from riglib.bmi.clda import CLDARecomputeParameters

class PlanarMultiLinkJointGoalCalculator(CLDARecomputeParameters):
    def calc(self, kin_chain, endpt_location):
        return kin_chain.inverse_kinematics(endpt_location)


class PlanarMultiLinkJointGoal(object):
    def __init__(self, ssm, shoulder_anchor, kin_chain):
        self.ssm = ssm
        self.shoulder_anchor = shoulder_anchor
        self.kin_chain = kin_chain
        self.prev_target_pos = None
        self.prev_target_state = None
        self.queued_target_pos = None

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.thread = PlanarMultiLinkJointGoalCalculator(input_queue, output_queue)
        self.waiting = False

    def __call__(self, target_pos):
        if target_pos == self.queued_target_pos:
            try:
                joint_pos = self.clda_output_queue.get_nowait()
                target_state = np.hstack([joint_pos, np.zeros_like(joint_pos), 1])

                # update the cache
                self.prev_target_pos = self.queued_target_pos
                self.prev_target_state = target_state
                return target_state
            except Queue.Empty:
                return self.prev_target_state
            except:
                pass
        elif not target_pos == self.prev_target_pos:
            endpt_location = target_pos - self.shoulder_anchor
            ik_data = (self.kin_chain, endpt_location)
            self.thread.put(ik_data)
            self.queued_target_pos = target_pos
            return self.prev_target_state
        else:
            return self.prev_target_state
