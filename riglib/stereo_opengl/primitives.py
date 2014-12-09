'''
Basic OpenGL shapes constructed out of triangular meshes
'''

import numpy as np
from numpy import pi
try:
    import pygame
except:
    import warnings
    warnings.warn('riglib/stereo_opengl_primitives.py: not importing name pygame')

from models import TriMesh

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
    def __init__(self, target_rad=4, segments=36, **kwargs):
        self.target_length = target_rad
        self.seg_per_side = segments/4
        zvals = self.target_length * np.linspace(-1, 1, num=segments)
        linevals = self.target_length * np.linspace(-1, 1, num=segments/4, endpoint=False)
        vertices = np.zeros(((len(zvals)) * len(linevals)* 4, 3))

        for i, z in enumerate(zvals):
            squarepoints = np.zeros((segments, 3))
            squarepoints[:,2] = z

            squarepoints[:self.seg_per_side,0] = linevals
            squarepoints[:self.seg_per_side,1] = np.ones((len(linevals), ))

            squarepoints[self.seg_per_side:2*self.seg_per_side,0] = np.ones((len(linevals), ))
            squarepoints[self.seg_per_side:2*self.seg_per_side,1] = linevals

            squarepoints[2*self.seg_per_side:3*self.seg_per_side,0] = linevals
            squarepoints[2*self.seg_per_side:3*self.seg_per_side,1] = -1*np.ones((len(linevals), ))

            squarepoints[3*self.seg_per_side:4*self.seg_per_side,0] = -1*np.ones((len(linevals), ))
            squarepoints[3*self.seg_per_side:4*self.seg_per_side,1] = linevals

            #r = np.sqrt(radius**2 - z**2)
            squarepoints[:,0] = self.target_length*squarepoints[:,0]
            squarepoints[:,1] = self.target_length*squarepoints[:,0]

            vertices[segments*i:segments*(i+1),:] = squarepoints
        
        vertices = np.vstack([vertices,(0,0,self.target_length),(0,0,-1*self.target_length)])
        rads = np.tile( np.array([np.sqrt(np.sum(np.square(vertices), axis = 1))]).T, (1, 3))
        allpointinds = np.arange(len(vertices))
        
        ####
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
        
        normals = np.divide(vertices, rads)
        hcoord = np.arctan2(normals[:,1], normals[:,0])
        vcoord = np.arctan2(normals[:,2], np.sqrt(vertices[:,0]**2 + vertices[:,1]**2))
        tcoord = np.array([(hcoord+pi) / (2*pi), (vcoord+pi/2) / pi]).T

        super(Cube, self).__init__(vertices, np.array(triangles), 
            tcoords=tcoord, normals=normals, **kwargs)

class Cylinder(TriMesh):
    def __init__(self, height=1, radius=1, segments=36, **kwargs):
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


class Circle(Shape2D):
    def __init__(self, center_pos, radius, *args, **kwargs):
        super(Circle, self).__init__(*args, **kwargs)
        self.center_pos = center_pos
        self.radius     = radius

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple(map(lambda x: int(255*x), self.color[0:3]))

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
            color = tuple(map(lambda x: int(255*x), self.color[0:3]))
            
            arc_angles = np.linspace(self.ang_range[0], self.ang_range[1], 5)
            pts = list(self.center_pos + self.radius*np.c_[np.cos(arc_angles), np.sin(arc_angles)])
            pts.append(self.center_pos)
            
            point_list = map(pos2pix_fn, pts)
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
            color = tuple(map(lambda x: int(255*x), self.color[0:3]))

            # create points and then rotate to correct orientation
            pts = np.array([[          0,  self.width/2], 
                            [          0, -self.width/2], 
                            [self.length, -self.width/2], 
                            [self.length,  self.width/2]])
            rot_mat = np.array([[np.cos(self.angle), -np.sin(self.angle)], 
                                [np.sin(self.angle),  np.cos(self.angle)]])
            pts = np.dot(rot_mat, pts.T).T + self.start_pos
            
            point_list = map(pos2pix_fn, pts)
            pygame.draw.polygon(surface, color, point_list)

        return self.visible  # return True if object was drawn
