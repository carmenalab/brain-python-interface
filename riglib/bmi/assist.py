import numpy as np 
from riglib.stereo_opengl import ik

def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
    diff_vec = target_pos - cursor_pos 
    dist_to_target = np.linalg.norm(diff_vec)
    dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
    
    if dist_to_target > target_radius:
        assist_cursor_pos = cursor_pos + speed*dir_to_target
    else:
        assist_cursor_pos = cursor_pos + speed*diff_vec/2

    assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
    Bu = assist_level * np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
    Bu = np.mat(Bu.reshape(-1,1))
    return Bu

def joint_5DOF_assist_endpoint_target(cursor_pos, target_pos, arm, **kwargs):
    assist_level = kwargs.pop('assist_level', 0.)

    # Get the endpoint control under full assist
    Bu_endpoint = endpoint_assist_simple(cursor_pos, target_pos, assist_level=1, **kwargs)

    # Convert the endpoint assist to joint space using IK/Jacobian
    Bu_endpoint = np.array(Bu_endpoint).ravel()
    endpt_pos = Bu_endpoint[0:3]
    endpt_vel = Bu_endpoint[3:6]

    # TODO when the arm configuration changes, these need to be switched!
    l_forearm, l_upperarm = arm.link_lengths
    shoulder_center = arm.xfm.move
    joint_pos, joint_vel = ik.inv_kin_2D(endpt_pos - shoulder_center, l_upperarm, l_forearm, vel=endpt_vel)

    Bu_joint = np.hstack([joint_pos[0].view((np.float64, 5)), joint_vel[0].view((np.float64, 5)), 1]).reshape(-1, 1)

    # Downweight the joint assist
    Bu = assist_level * np.mat(Bu_joint).reshape(-1,1)
    return Bu