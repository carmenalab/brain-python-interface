'''
Quaternions and generic 3D transformations
'''



import numpy as np

class Quaternion(object):
    def __init__(self, w=1, x=0, y=0, z=0):
        if isinstance(w, (list, tuple, np.ndarray)) and not isinstance(x, np.ndarray):
            #Allows the use of a quaternion as a vector
            self.quat = np.array([0, w[0], w[1], w[2]])
        else:
            self.quat = np.array([w, x, y, z])
    
    def __repr__(self):
        if self.quat.ndim > 1:
            return "<Quaternion set for %d rotations>"%self.quat.shape[1]
        return "%g+%gi+%gj+%gk"%tuple(self.quat)
    
    def norm(self):
        self.quat = self.quat / np.sqrt((self.quat**2).sum())
        return self
    
    def conj(self):
        return Quaternion(self.w, *(-self.vec))

    @property
    def H(self):
        return self.conj()
    
    def __getattr__(self, attr):
        if attr in ["w", "scalar"]:
            return self.quat[0]
        elif attr in ["x", "i"]:
            return self.quat[1]
        elif attr in ["y", "j"]:
            return self.quat[2]
        elif attr in ["z", "k"]:
            return self.quat[3]
        elif attr in ["v", "vec", "vector"]:
            return self.quat[1:]
        else:
            super(Quaternion, self).__getattr__(self, attr)
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w*other.w   - (self.vec*other.vec).sum(0)
            v = self.w*other.vec + other.w*self.vec + np.cross(self.vec.T, other.vec.T).T
            return Quaternion(w, *v).norm()
        elif isinstance(other, (np.ndarray, list, tuple)):
            if isinstance(other, (list, tuple)):
                other = np.array(other)
            #rotate a vector, will need to be implemented in GLSL eventually
            cross = np.cross(self.vec.T, other) + self.w*other
            return (other + np.cross(2*self.vec.T, cross)).squeeze()
            '''
            conj = self.H
            w = -np.dot(other, conj.vec)
            vec = np.outer(conj.w, other) + np.cross(other, conj.vec.T)
            if self.quat.ndim > 1:
                return self.w*vec.T + np.
            return self.w*vec + np.outer(w, self.vec).squeeze() + np.cross(self.vec, vec)
            '''
        else:
            raise ValueError

    def to_mat(self):
        '''
        Convert to an augmented rotation matrix if the quaternion is of unit norm
        ??? Does this function provide a sensible result for non-unit quaternions?

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of shape (4, 4)
            Affine transformation matrix
        '''
        a, b, c, d = self.quat
        return np.array([
            [a**2+b**2-c**2-d**2,       2*b*c-2*a*d,            2*b*d+2*a*c,        0],
            [   2*b*c+2*a*d,       a**2-b**2+c**2-d**2,         2*c*d-2*a*b,        0],
            [   2*b*d-2*a*c,            2*c*d+2*a*b,        a**2-b**2-c**2+d**2,    0],
            [   0,                      0,                          0,              1]])

    @classmethod 
    def from_mat(cls, M):
        qw = np.sqrt(1 + M[0,0] + M[1,1] + M[2,2]) / 2
        qx = (M[2,1] - M[1,2])/(4*qw)
        qy = (M[0,2] - M[2,0])/(4*qw)
        qz = (M[1,0] - M[0,1])/(4*qw)
        return Quaternion(w=qw, x=qx, y=qy, z=qz)

    def rotate_to(self, vec):
        svec = self.vec / np.sqrt((self.vec**2).sum())
        nvec = nvec = vec2 / np.sqrt((vec2**2).sum())
        rad = np.arccos(np.dot(svec, nvec))
        axis = np.cross(svec, nvec)
        self = self.from_axisangle(axis, rad)*self

    @classmethod
    def rotate_vecs(cls, vec1, vec2):
        '''
        Get the quaternion which rotates vec1 onto vec2

        Parameters
        ----------
        vec1: np.ndarray of shape (3,)
            Starting vector
        vec2: np.ndarray of shape (3,)
            Vector which defines the orientation that you want to rotate the first vector to

        Returns
        -------
        Quaternion representing the rotation
        '''
        vec1, vec2 = np.array(vec1), np.array(vec2)
        svec = vec1 / np.sqrt((vec1**2).sum())
        nvec = vec2 / np.sqrt((vec2**2).sum())
        if nvec.ndim > 1:
            if svec.ndim > 1:
                rad = (svec * nvec).sum(1)
            else:
                rad = np.arccos(np.dot(svec, nvec.T))
        else:
            rad = np.arccos(np.dot(svec, nvec))
        axis = np.cross(svec, nvec)
        return cls.from_axisangle(axis, rad)
    
    @classmethod
    def from_axisangle(cls, axis, rad):
        '''
        Convert from the Axis-angle representation of rotations to the quaternion representation

        Parameters
        ----------
        axis: np.ndarray of shape (3,) or ?????
            Rotation axis
        rad: float
            Angle to rotate around the specified axis in radians

        Returns
        -------
        Quaternion representing the rotation
        '''
        #normalize the axis first
        axis = np.array(axis)
        if axis.ndim > 1:
            axis = axis.T / np.sqrt((axis**2).sum(1))
        else:
            if not np.all(axis == 0):
                axis = axis / np.sqrt((axis**2).sum())
        w = np.cos(rad*0.5)
        v = axis * np.sin(rad*0.5)
        return cls(w, *v)

class Transform(object):
    '''
    Homogenous transformations ???
    '''
    def __init__(self, move=(0,0,0), scale=1, rotate=None):
        self.move = np.array(move, dtype=float)
        self.scale = scale
        self.rotate = rotate if rotate is not None else Quaternion()

    def __repr__(self):
        return "Rotate %s, then scale %s, then translate %s"%(self.rotate, self.scale, self.move)
    
    def __mul__(self, other):
        if isinstance(other, Transform):
            #Pre-multiply the other transform, then apply self
            move = self.move + self.rotate*other.move
            scale = self.scale * other.scale
            rot = self.rotate * other.rotate
            return Transform(move, scale, rot)

        elif isinstance(other, Quaternion):
            #Apply the quaternion directly to current rotation
            return Transform(self.move, self.scale, other.rotate * self.rotate)

    def __call__(self, vecs):
        return self.scale * (self.rotate * vecs) + self.move

    def translate(self, x, y, z, reset=False):
        '''
        Set the translation point of the transformation

        Parameters
        ----------
        x, y, z: float
            Coordinates representing how much to move
        reset: bool, optional, default=False 
            If true, the new coordinates replace the old ones. If false, they are added on
        '''
        if reset:
            self.move[:] = x,y,z
        else:
            self.move += x,y,z
        return self

    def rotate_x(self, rad, reset=False):
        rotate = Quaternion.from_axisangle((1,0,0), rad)
        if reset:
            self.rotate = rotate
        else:
            self.rotate = (rotate * self.rotate).norm()
        return self

    def rotate_y(self, rad, reset=False):
        rotate = Quaternion.from_axisangle((0,1,0), rad)
        if reset:
            self.rotate = rotate
        else:
            self.rotate = (rotate * self.rotate).norm()
        return self

    def rotate_z(self, rad, reset=False):
        rotate = Quaternion.from_axisangle((0,0,1), rad)
        if reset:
            self.rotate = rotate
        else:
            self.rotate = (rotate * self.rotate).norm()
        return self
    
    def to_mat(self):
        scale = np.eye(4)
        scale[(0,1,2), (0,1,2)] = self.scale
        move = np.eye(4)
        move[:3, -1] = self.move
        
        return np.dot(move, np.dot(scale, self.rotate.to_mat()))

def test():
    world = Transform().rotate_x(np.radians(-90))
    eye = Transform().translate(0,35,0)
    obj = Transform().translate(0,10,5)
    assert np.allclose((world*eye*obj)((0,0,1)), [0,6,-45])
    obj.rotate_y(np.radians(-90))
    assert np.allclose((world*eye*obj)((0,0,1)), [-1, 5, -45])
    obj.rotate_z(np.radians(-90))
    assert np.allclose((world*eye*obj)((0,0,1)), [0,5,-46])
    assert np.allclose(np.dot((world*eye*obj).to_mat(), [0,0,1,1]), [0,5,-46, 1])
