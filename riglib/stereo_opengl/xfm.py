from __future__ import division
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
        self.quat /= np.sqrt((self.quat**2).sum())
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
        raise ValueError

    def to_mat(self):
        a, b, c, d = self.quat
        return np.array([
            [a**2+b**2-c**2-d**2,       2*b*c-2*a*d,            2*b*d+2*a*c,        0],
            [   2*b*c+2*a*d,       a**2-b**2+c**2-d**2,         2*c*d-2*a*b,        0],
            [   2*b*d-2*a*c,            2*c*d+2*a*b,        a**2-b**2-c**2+d**2,    0],
            [   0,                      0,                          0,              1]])

    def rotate_to(self, vec):
        svec = self.vec / np.sqrt((self.vec**2).sum())
        nvec = nvec = vec2 / np.sqrt((vec2**2).sum())
        rad = np.arccos(np.dot(svec, nvec))
        axis = np.cross(svec, nvec)
        self = self.from_axisangle(axis, rad)*self

    @classmethod
    def rotate_vecs(cls, vec1, vec2):
        '''Get the quaternion which rotates a vector onto another one'''
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
        #normalize the axis first
        axis = np.array(axis)
        if axis.ndim > 1:
            axis = axis.T / np.sqrt((axis**2).sum(1))
        else:
            axis /= np.sqrt((axis**2).sum())
        w = np.cos(rad*0.5)
        v = axis * np.sin(rad*0.5)
        return cls(w, *v)

class Transform(object):
    def __init__(self, move=(0,0,0), scale=1, rotate=None):
        self.move = np.array(move, dtype=np.float)
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
