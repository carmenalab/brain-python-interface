'''Needs docs'''

import numpy as np
from numpy import pi

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


## below code added by Sid, only for use with WindowDispl2D class (not Window)
class Sector(object):
    def __init__(self, center_pos, radius, ang_range, color):
        self.center_pos = center_pos
        self.radius     = radius
        self.ang_range  = ang_range
        self.color      = color

class Line(object):
    def __init__(self, start_pos, length, angle, width, color):
        self.start_pos = start_pos
        self.length    = length
        self.angle     = angle
        self.width     = width
        self.color     = color