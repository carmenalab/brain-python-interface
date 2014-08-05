from models import Group, GroupDispl2D
from xfm import Quaternion
from riglib.stereo_opengl.primitives import Sphere, Cylinder


class Box(Group):
    def __init__(self, **kwargs):
        bcolor = (181/256., 116/256., 96/256., 1)
        sidelen = 16
        linerad=.1
        self.vert_box = Group([
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, -sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, -sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, sidelen/2, -sidelen/2)])
        self.hor_box = Group([
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, -sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, -sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, sidelen/2, -sidelen/2)])
        self.depth_box = Group([
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, -sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, -sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, sidelen/2, -sidelen/2),
            Cylinder(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, sidelen/2, -sidelen/2)])
        self.hor_box.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (1,0,0))
        self.depth_box.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (0,1,0))
        self.box = Group([self.hor_box, self.depth_box, self.vert_box])
        super(Box, self).__init__([self.box], **kwargs)