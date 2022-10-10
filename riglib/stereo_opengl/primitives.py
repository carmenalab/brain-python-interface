'''
Basic OpenGL shapes constructed out of triangular meshes
'''

import numpy as np
from numpy import pi
try:
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"    
    import pygame
except:
    import warnings
    warnings.warn('riglib/stereo_opengl_primitives.py: not importing name pygame')

from .models import TriMesh

class Plane(TriMesh):
    def __init__(self, width=1, height=1, **kwargs):
        pts = np.array([[0,0,0],
                        [width,0,0],
                        [width,height,0],
                        [0,height,0]])
        polys = [(0,1,3),(1,2,3)]
        tcoords = np.array([[0,0],[1,0],[1,1],[0,1]])
        normals = [(0,0,1)]*4
        super(Plane, self).__init__(pts, np.array(polys), 
                tcoords=tcoords, normals=np.array(normals), **kwargs)

class Cube(TriMesh):
    def __init__(self, side_len=1, side_height=None, segments=36, **kwargs):
        self.side_len = side_len
        if side_height is None:
            side_len_half = side_len/2.
        else:
            side_len_half = side_height # 0.5
        side = np.linspace(-1, 1, int(segments/4), endpoint=True)
        
        unit1 = np.hstack(( side[:,np.newaxis], np.ones((len(side),1)), np.ones((len(side),1)) ))
        unit2 = np.hstack(( np.ones((len(side),1)), side[::-1,np.newaxis], np.ones((len(side),1)) ))
        unit3 = np.hstack(( side[::-1,np.newaxis], -1*np.ones((len(side),1)), np.ones((len(side),1)) ))
        unit4 = np.hstack(( -1*np.ones((len(side),1)), side[:,np.newaxis], np.ones((len(side),1)) ))

        unit = np.vstack((unit1, unit2, unit3, unit4))

        pts = np.vstack([unit*[side_len_half, side_len_half, 0], unit*[side_len_half,side_len_half,side_len]])
        normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])
        #pts = np.vstack([unit*[side_len, 0, 0], unit*[side_len,0,side_height]])
        #normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])
        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super(Cube, self).__init__(pts, np.array(polys), 
            tcoords=tcoord, normals=normals, **kwargs)

class Cylinder(TriMesh):
    def __init__(self, height=1, radius=1, segments=36, **kwargs):
        self.height = height
        self.radius = radius
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        unit = np.array([np.cos(theta), np.sin(theta), np.ones(segments)]).T

        pts = np.vstack([unit*[radius, radius, 0], unit*[radius,radius,height]])
        normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])

        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super(Cylinder, self).__init__(pts, np.array(polys), 
            tcoords=tcoord, normals=normals, **kwargs)

class Cable(TriMesh):
    def __init__(self,radius=.5, trajectory = np.array([np.sin(x) for x in range(60)]), segments=12,**kwargs):
        self.trial_trajectory = trajectory
        self.center_value = [0,0,0]
        self.radius = radius
        self.segments = segments
        self.update(**kwargs)
    
    def update(self, **kwargs):
        theta = np.linspace(0, 2*np.pi, self.segments, endpoint=False)
        unit = np.array([np.ones(self.segments),np.cos(theta) ,np.sin(theta)]).T
        intial = np.array([[0,0,self.trial_trajectory[x]] for x in range(len(self.trial_trajectory))])
        self.pts = (unit*[-30/1.36,self.radius,self.radius])+intial[0]
        for i in range(1,len(intial)):
            self.pts = np.vstack([self.pts, (unit*[(i-30)/3,self.radius,self.radius])+intial[i]])

        self.normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])
        self.polys = []
        for i in range(self.segments-1):
            for j in range(len(intial)-1): 
                self.polys.append((i+j*self.segments, i+1+j*self.segments, i+self.segments+j*self.segments))
                self.polys.append((i+self.segments+j*self.segments, i+1+j*self.segments, i+1+self.segments+j*self.segments))

        tcoord = np.array([np.arange(self.segments), np.ones(self.segments)]).T
        n = 1./self.segments
        self.tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])
        super(Cable, self).__init__(self.pts, np.array(self.polys), 
            tcoords=self.tcoord, normals=self.normals, **kwargs)


class Sphere(TriMesh):
    def __init__(self, radius=1, segments=36, **kwargs):
        self.radius = radius
        zvals = radius * np.cos(np.linspace(0, np.pi, num=segments))
        circlevals = np.linspace(0, 2*pi, num=segments, endpoint=False)

        vertices = np.zeros(((len(zvals)-2) * len(circlevals), 3))

        for i, z in enumerate(zvals[1:-1]):
            circlepoints = np.zeros((segments, 3))
            circlepoints[:,2] = z
            r = np.sqrt(radius**2 - z**2)
            circlepoints[:,0] = r*np.sin(circlevals)
            circlepoints[:,1] = r*np.cos(circlevals)
            vertices[segments*i:segments*(i+1),:] = circlepoints
        
        vertices = np.vstack([vertices,(0,0,radius),(0,0,-radius)])
        allpointinds = np.arange(len(vertices))
        
        triangles = np.zeros((segments,3))
        firstcirc = allpointinds[0:segments]
        triangles[0,:] = (allpointinds[-2],firstcirc[0], firstcirc[-1])
        for i in range(segments-1):
            triangles[i+1,:] = (allpointinds[-2], firstcirc[i+1], firstcirc[i])
        
        triangles = list(triangles)
        for i in range(segments-3):
            points1 = allpointinds[i*segments:(i+1)*segments]
            points2 = allpointinds[(i+1)*segments:(i+2)*segments]
            for ind, p in enumerate(points1[:-1]):
                t1 = (p, points1[ind+1], points2[ind+1])
                t2 = (p, points2[ind+1], points2[ind])
                triangles += [t1, t2]
            triangles += [(points1[-1], points1[0], points2[0]), (points1[-1], points2[0], points2[-1])]
        
        bottom = np.zeros((segments,3))
        lastcirc = allpointinds[-segments-2:-2]
        bottom[0,:] = (allpointinds[-1], lastcirc[-1], lastcirc[0]) 
        for i in range(segments-1):
            bottom[i+1,:] = (allpointinds[-1], lastcirc[i], lastcirc[i+1])
        triangles = np.vstack([triangles, bottom])
        
        normals = vertices/radius
        hcoord = np.arctan2(normals[:,1], normals[:,0])
        vcoord = np.arctan2(normals[:,2], np.sqrt(vertices[:,0]**2 + vertices[:,1]**2))
        tcoord = np.array([(hcoord+pi) / (2*pi), (vcoord+pi/2) / pi]).T

        super(Sphere, self).__init__(vertices, np.array(triangles), 
            tcoords=tcoord, normals=normals, **kwargs)


class Cone(TriMesh):
    def __init__(self, height=1, radius1=1, radius2=1, segments=36, **kwargs):
        self.height = height
        self.radius1 = radius1
        self.radius2 = radius2
        self.radius = radius1 # for pretending it's a cylinder..
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        unit = np.array([np.cos(theta), np.sin(theta), np.ones(segments)]).T

        pts = np.vstack([unit*[radius1, radius1, 0], unit*[radius2,radius2,height]])
        normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])

        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super(Cone, self).__init__(pts, np.array(polys), 
            tcoords=tcoord, normals=normals, **kwargs)


class Chain(object):
    '''
    An open chain of cylinders and cones, e.g. to simulate a stick-figure arm/robot
    '''
    def __init__(self, link_radii, joint_radii, link_lengths, joint_colors, link_colors):
        from .models import Group
        from .xfm import Quaternion
        self.num_joints = num_joints = len(link_lengths)

        self.link_radii = self.make_list(link_radii, num_joints)
        self.joint_radii = self.make_list(joint_radii, num_joints)
        self.link_lengths = self.make_list(link_lengths, num_joints)
        self.joint_colors = self.make_list(joint_colors, num_joints)
        self.link_colors = self.make_list(link_colors, num_joints)        

        self.links = []

        # Create the link graphics
        for i in range(self.num_joints):
            joint = Sphere(radius=self.joint_radii[i], color=self.joint_colors[i])

            # The most distal link gets a tapered cylinder (for purely stylistic reasons)
            if i < self.num_joints - 1:
                link = Cylinder(radius=self.link_radii[i], height=self.link_lengths[i], color=self.link_colors[i])
            else:
                link = Cone(radius1=self.link_radii[-1], radius2=self.link_radii[-1]/2, height=self.link_lengths[-1], color=self.link_colors[-1])
            link_i = Group((link, joint))
            self.links.append(link_i)

        link_offsets = [0] + self.link_lengths[:-1]
        self.link_groups = [None]*self.num_joints
        for i in range(self.num_joints)[::-1]:
            if i == self.num_joints-1:
                self.link_groups[i] = self.links[i]
            else:
                self.link_groups[i] = Group([self.links[i], self.link_groups[i+1]])

            self.link_groups[i].translate(0, 0, link_offsets[i])

    def _update_link_graphics(self, curr_vecs):
        from .models import Group
        from .xfm import Quaternion

        for i in range(self.num_joints):
            # Rotate each joint to the vector specified by the corresponding row in self.curr_vecs
            # Annoyingly, the baseline orientation of the first group is always different from the 
            # more distal attachments, so the rotations have to be found relative to the orientation 
            # established at instantiation time.
            if i == 0:
                baseline_orientation = (0, 0, 1)
            else:
                baseline_orientation = (1, 0, 0)

            # Find the normalized quaternion that represents the desired joint rotation
            self.link_groups[i].xfm.rotate = Quaternion.rotate_vecs(baseline_orientation, curr_vecs[i]).norm()

            # Recompute any cached transformations after the change
            self.link_groups[i]._recache_xfm()

    def translate(self, *args, **kwargs):
        self.link_groups[0].translate(*args, **kwargs)

    @staticmethod
    def make_list(value, num_joints):
        '''
        Helper function to allow joint/link properties of the chain to be specified
        as one value for all joints/links or as separate values for each
        '''
        if isinstance(value, list) and len(value) == num_joints:
            return value
        else:
            return [value] * num_joints

##### 2-D primitives #####

class Shape2D(object):
    '''Abstract base class for shapes that live in the 2-dimension xz-plane
    and are intended only for use with the WindowDispl2D class (not Window).
    '''

    def __init__(self, color, visible=True):
        self.color   = color
        self.visible = visible

    def draw(self, surface, pos2pix_fn):
        '''Draw itself on the given pygame.Surface object using the given
        position-to-pixel_position function.'''

        raise NotImplementedError  # implement in subclasses

    def _recache_xfm(self):
        pass


class Circle(Shape2D):
    def __init__(self, center_pos, radius, *args, **kwargs):
        super(Circle, self).__init__(*args, **kwargs)
        self.center_pos = center_pos
        self.radius     = radius

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple([int(255*x) for x in self.color[0:3]])

            pix_pos    = pos2pix_fn(self.center_pos)
            pix_radius = pos2pix_fn([self.radius, 0])[0] - pos2pix_fn([0, 0])[0]
            pygame.draw.circle(surface, color, pix_pos, pix_radius)

        return self.visible  # return True if object was drawn


class Sector(Shape2D):
    def __init__(self, center_pos, radius, ang_range, *args, **kwargs):
        super(Sector, self).__init__(*args, **kwargs)
        self.center_pos = center_pos
        self.radius     = radius
        self.ang_range  = ang_range

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple([int(255*x) for x in self.color[0:3]])
            
            arc_angles = np.linspace(self.ang_range[0], self.ang_range[1], 5)
            pts = list(self.center_pos + self.radius*np.c_[np.cos(arc_angles), np.sin(arc_angles)])
            pts.append(self.center_pos)
            
            point_list = list(map(pos2pix_fn, pts))
            pygame.draw.polygon(surface, color, point_list)
        
        return self.visible  # return True if object was drawn


class Line(Shape2D):
    def __init__(self, start_pos, length, width, angle, *args, **kwargs):
        super(Line, self).__init__(*args, **kwargs)
        self.start_pos = start_pos
        self.length    = length
        self.width     = width  # draw a line as thin rectangle
        self.angle     = angle

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple([int(255*x) for x in self.color[0:3]])

            # create points and then rotate to correct orientation
            pts = np.array([[          0,  self.width/2], 
                            [          0, -self.width/2], 
                            [self.length, -self.width/2], 
                            [self.length,  self.width/2]])
            rot_mat = np.array([[np.cos(self.angle), -np.sin(self.angle)], 
                                [np.sin(self.angle),  np.cos(self.angle)]])
            pts = np.dot(rot_mat, pts.T).T + self.start_pos
            
            point_list = list(map(pos2pix_fn, pts))
            pygame.draw.polygon(surface, color, point_list)

        return self.visible  # return True if object was drawn
