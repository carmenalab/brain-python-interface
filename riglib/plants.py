#!/usr/bin/python
'''
Representations of plants (control systems)
'''
import numpy as np
from stereo_opengl.primitives import Cylinder, Sphere, Cone

class Plant(object):
    def __init__(self):
        raise NotImplementedError

    def drive(self, decoder):
        self.set_intrinsic_coordinates(decoder['q'])
        intrinsic_coords = self.get_intrinsic_coordinates()
        if not np.any(np.isnan(intrinsic_coords)):
            decoder['q'] = self.get_intrinsic_coordinates()


class CursorPlant(Plant):
    def __init__(self, endpt_bounds=None, cursor_radius=0.4, cursor_color=(.5, 0, .5, 1), starting_pos=np.array([0., 0., 0.]), **kwargs):
        self.endpt_bounds = endpt_bounds
        self.position = starting_pos
        self.starting_pos = starting_pos
        self.cursor_radius = cursor_radius
        self.cursor_color = cursor_color
        self._pickle_init()

    def _pickle_init(self):
        self.cursor = Sphere(radius=self.cursor_radius, color=self.cursor_color)
        self.cursor.translate(*self.position, reset=True)
        self.graphics_models = [self.cursor]

    def get_endpoint_pos(self):
        return self.position

    def set_endpoint_pos(self, pt, **kwargs):
        self.position = pt

    def get_intrinsic_coordinates(self):
        return self.position

    def set_intrinsic_coordinates(self, pt):
        self.position = pt

    def drive(self, decoder):
        pos = decoder['q'].copy()
        vel = decoder['qdot'].copy()
        
        if self.endpt_bounds is not None:
            if pos[0] < self.endpt_bounds[0]: 
                pos[0] = self.endpt_bounds[0]
            if pos[0] > self.endpt_bounds[1]: 
                pos[0] = self.endpt_bounds[1]

            if pos[1] < self.endpt_bounds[2]: 
                pos[1] = self.endpt_bounds[2]
            if pos[1] > self.endpt_bounds[3]: 
                pos[1] = self.endpt_bounds[3]

            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
        
        decoder['q'] = pos
        decoder['qdot'] = vel
        super(CursorPlant, self).drive(decoder)



class VirtualKinematicChain(Plant):
	pass
