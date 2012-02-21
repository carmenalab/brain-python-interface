import numpy as np

from models import TriMesh

class Cylinder(TriMesh):
    def __init__(self, ctx, height=1, radius=1, segments=36, xfm=np.eye(4)):
        theta = np.linspace(0, 2*np.pi, segments)
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

        super(Cylinder, self).__init__(ctx, pts, polys, tcoords=tcoord, xfm=xfm)