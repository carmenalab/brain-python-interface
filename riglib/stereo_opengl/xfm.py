import numpy as np

class Quaternion(object):
    def __init__(self, w=1, x=0, y=0, z=0):
        self.quat = np.array([w, x, y, z])
    
    def __str__(self):
        return "%f+%fi+%fj+%fk"%self.quat
    
    def norm(self):
        self.quat /= np.sqrt((self.quat**2).sum())
    
    def conj(self):
        return Quaternion(w, -x, -y, -z)
    
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
    
    def __mult__(self, other):
        if isinstance(other, Quaternion):
            w = self.w*other.w   - np.dot(self.vec, other.vec)
            v = self.w*other.vec + other.w*self.vec + np.cross(self.vec, other.vec)
            return Quaternion(w, *v)
        elif isinstance(other, np.ndarray):
            #rotate a vector, will need to be implemented in GLSL eventually
            conj = self.conj
            w = -np.dot(other, conj.vec)
            vec = conj.w*other + np.cross(other, conj.vec)
            nw = self.w*w - np.dot(self.vec, vec)
            pts = self.w*vec + w*self.vec + np.cross(self.vec, vec)
            return nw, pts
    
    @classmethod
    def from_axisangle(cls, axis, angle):
        #normalize the axis first
        axis /= np.sqrt((axis**2).sum())
        w = np.cos(angle*0.5)
        v = axis / np.sqrt((axis**2).sum()) * np.sine(angle*0.5)
        return cls(w, *v)

class Transform(object):
    def __init__(self, move=(0,0,0), scale=1, rotate=None):
        self.move = np.array(move)
        self.scale = scale
        self.rotate = rotate if rotate is not None else Quaternion()
    
    def __mult__(self, other):
        assert isinstance(other, Transform)
        move = other.rotate*self.move + other.move
        scale = self.scale * other.scale
        rot = self.rotate * other.rotate
        return Transform(move, scale, rot)
    
    def to_mat(self):
        a = self.rot.w
        b, c, d = self.rot.vec

        rot = np.array([
            [a**2+b**2-c**2-d**2,       2*b*c-2*a*d,            2*b*d+2*a*c,        0],
            [   2*b*c+2*a*d,       a**2-b**2+c**2-d**2,         2*c*d-2*a*b,        0],
            [   2*b*d-2*a*c,            2*c*d+2*a*b,        a**2-b**2-c**2+d**2,    0],
            [   0,                      0,                          0,              1]])
        scale = np.eye(4)
        scale[(0,0), (1,1), (2,2)] = self.scale
        move = np.eye(4)
        move[:3, -1] = self.move

        return np.dot(move, np.dot(scale, rot))
