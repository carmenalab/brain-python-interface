'''
List of possible "plants" that a subject could control either during manual or brain control
'''
import numpy as np
from riglib import plants

pi = np.pi
RED = (1,0,0,.5)
## BMI Plants
cursor_14x14 = plants.CursorPlant(endpt_bounds=(-14, 14, 0., 0., -14, 14))

shoulder_anchor = np.array([2., 0., -15])
chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))

chain_20_20_endpt = plants.EndptControlled2LArm(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
init_pos = np.array([0, 0, 0], np.float64)
chain_20_20_endpt.set_intrinsic_coordinates(init_pos)

chain_20_20 = plants.RobotArmGen2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
init_pos = np.array([ 0.38118002,  2.08145271])
chain_20_20.set_intrinsic_coordinates(init_pos)

plantlist = dict(
	cursor_14x14=cursor_14x14, 
	chain_20_20=chain_20_20, 
	chain_20_20_endpt=chain_20_20_endpt)