import numpy as np

from models import TriMesh

class Cylinder(TriMesh):
    def __init__(self, height=1, radius=1, segments=36, xfm=np.eye(4), shader="default", color=(0.5,0.5,0.5,1)):
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        unit = np.array([np.cos(theta), np.sin(theta), np.ones(segments)]).T

        pts = np.vstack([unit*[radius, radius, 0], unit*[radius,radius,height]])
        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super(Cylinder, self).__init__(pts, np.array(polys), tcoords=tcoord, xfm=xfm, shader=shader, color=color)

class Plane(TriMesh):
    def __init__(self, width=1, height=1, xfm=np.eye(4), color=(0.5,0.5,0.5,1)):
        pts = np.array([[0,0,0],
                        [width,0,0],
                        [width,height,0],
                        [0,height,0]])
        polys = [(0,1,3),(1,2,3)]
        tcoords = np.array([[0,0],[1,0],[1,1],[0,1]])
        super(Plane, self).__init__(pts, np.array(polys), tcoords=tcoords, xfm=xfm, shader="default", color=color)