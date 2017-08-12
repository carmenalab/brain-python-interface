'''
List of possible "plants" that a subject could control either during manual or brain control
'''
import numpy as np
from riglib import plants

pi = np.pi
RED = (1,0,0,.5)
## BMI Plants
cursor_14x14 = plants.CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14))
cursor_25x14 = plants.CursorPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14))
big_cursor_25x14 = plants.CursorPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14), cursor_radius=1.0)

cursor_14x14_no_vel_wall = plants.CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14), vel_wall=False)

chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))
shoulder_anchor = np.array([2., 0., -15])
chain_15_15_5_5 = plants.RobotArmGen2D(link_lengths=[15, 15, 5, 5], base_loc=shoulder_anchor, **chain_kwargs)
init_joint_pos = np.array([ 0.47515737,  1.1369006 ,  1.57079633,  0.29316668])
chain_15_15_5_5.set_intrinsic_coordinates(init_joint_pos)

tentacle2 = plants.RobotArmGen2D(link_lengths=[10, 10, 10, 10], base_loc=shoulder_anchor, **chain_kwargs)
init_joint_pos = np.array([ 0.47515737,  1.1369006 ,  1.57079633,  0.29316668])
tentacle2.set_intrinsic_coordinates(init_joint_pos)

chain_15_15_5_5_on_screen = plants.RobotArmGen2D(link_lengths=[15, 15, 5, 5], base_loc=shoulder_anchor, stay_on_screen=True, **chain_kwargs)
chain_15_15_5_5_on_screen.set_intrinsic_coordinates(init_joint_pos)


chain_20_20_endpt = plants.EndptControlled2LArm(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
init_pos = np.array([0, 0, 0], np.float64)
chain_20_20_endpt.set_intrinsic_coordinates(init_pos)

chain_20_20 = plants.RobotArmGen2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
init_pos = np.array([pi/2, pi/2])
init_pos = np.array([ 0.38118002,  2.08145271])
chain_20_20.set_intrinsic_coordinates(init_pos)

cursor_onedimLFP = plants.onedimLFP_CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14), lfp_cursor_rad=1.5, lfp_cursor_color=(248/256., 220/256., 0/256., 1))
cursor_2dimLFP = plants.twodimLFP_CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14), lfp_cursor_rad=1.5, lfp_cursor_color=(248/256., 220/256., 0/256., 1))

#cursor_onedimLFP_or = plants.onedimLFP_CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14), lfp_cursor_rad=1.5, lfp_cursor_color=(250/256., 102/256., 0/256., 1))
inv_cursor_onedimLFP = plants.onedimLFP_CursorPlant_inverted(endpt_bounds=(-14, 14, 0., 0., -14, 14), lfp_cursor_rad=1.5, lfp_cursor_color=(248/256., 220/256., 0/256., 1))

#ratBMI cursor
min_freq = 400.
max_freq = 15000.
#rat_cursor = plants.AuditoryCursor(min_freq, max_freq, sound_duration=0.1)


plantlist = dict(
	cursor_14x14=cursor_14x14, 
	cursor_25x14=cursor_25x14, 
	big_cursor_25x14 = big_cursor_25x14,
	chain_15_15_5_5=chain_15_15_5_5, 
	chain_15_15_5_5_on_screen=chain_15_15_5_5_on_screen, 
	chain_20_20=chain_20_20, 
	chain_20_20_endpt=chain_20_20_endpt, 
	cursor_onedimLFP=cursor_onedimLFP, 
    cursor_2dimLFP=cursor_2dimLFP,
    inv_cursor_onedimLFP=inv_cursor_onedimLFP,
    tentacle2=tentacle2)
    #aud_cursor=rat_cursor)

