'''
Graphical display classes. Experimental tasks involving graphical displays 
inherit from these classes.
'''


from __future__ import division
import os

import numpy as np
from OpenGL.GL import *

from riglib.experiment import LogExperiment
from riglib.experiment import traits

from render import stereo
from models import Group, GroupDispl2D
from xfm import Quaternion
from riglib.stereo_opengl.primitives import Sphere
from riglib.stereo_opengl.environment import Box
import time
from config import config
from primitives import Cylinder, Sphere, Cone
from profile_support import profile

try:
    import pygame
except:
    import warnings
    warnings.warn('riglib/stereo_opengl/window.py: not importing name pygame')

# for WindowDispl2D only
from riglib.stereo_opengl.primitives import Shape2D


class Window(LogExperiment):
    status = dict(draw=dict(stop=None))
    state = "draw"
    stop = False

    window_size = (3840, 1080)
    background = (0,0,0,1)
    # fps = 60  # TODO (already defined in Experiment class)

    #Screen parameters, all in centimeters -- adjust for monkey
    fov = np.degrees(np.arctan(14.65/(44.5+3)))*2
    screen_dist = 44.5+3
    iod = 2.5

    show_environment = traits.Int(0)

    def __init__(self, **kwargs):
        # self.window_size = (self.window_size[0]*2, self.window_size[1]) # Stereo window display
        super(Window, self).__init__(**kwargs)

        self.models = []
        self.world = None
        self.event = None

        # os.popen('sudo vbetool dpms on')

        if self.show_environment:
            self.add_model(Box())

    def screen_init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = config.display_start_pos
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()
        
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL | pygame.NOFRAME
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS,1)
            self.surf = pygame.display.set_mode(self.window_size, flags)
        except:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS,0)
            self.surf = pygame.display.set_mode(self.window_size, flags)

        glEnable(GL_BLEND)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_TEXTURE_2D)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.background)
        glClearDepth(1.0)
        glDepthMask(GL_TRUE)
        
        self.renderer = self._get_renderer()
        
        #this effectively determines the modelview matrix
        self.world = Group(self.models)
        self.world.init()

        #up vector is always (0,0,1), why would I ever need to roll the camera?!
        self.set_eye((0,-self.screen_dist,0), (0,0))
    
    def _get_renderer(self):
        return stereo.MirrorDisplay(self.window_size, self.fov, 1, 1024, self.screen_dist, self.iod)
    
    def set_eye(self, pos, vec, reset=True):
        '''Set the eye's position and direction. Camera starts at (0,0,0), pointing towards positive y'''
        self.world.translate(pos[0], pos[2], pos[1], reset=True).rotate_x(-90)
        self.world.rotate_y(vec[0]).rotate_x(vec[1])

    def add_model(self, model):
        if self.world is None:
            #world doesn't exist yet, add the model to cache
            self.models.append(model)
        else:
            #We're already running, initialize the model and add it to the world
            model.init()
            self.world.add(model)

    @property 
    def update_rate(self):
        '''
        Attribute for update rate of task. Using @property in case any future modifications
        decide to change fps on initialization
        '''
        return 1./self.fps

    def show_object(self, obj, show=False):
        '''
        Show or hide an object
        '''
        if show:
            obj.attach()
        else:
            obj.detach()
    
    def draw_world(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.renderer.draw(self.world)
        pygame.display.flip()
        self.renderer.draw_done()
    
    def _get_event(self):
        for e in pygame.event.get(pygame.KEYDOWN):
            return (e.key, e.type)
    
    def _start_None(self):
        pygame.display.quit()

    def _start_reward(self):
        pass

    def _start_wait(self):
        pass
    
    def _test_stop(self, ts):
        return self.stop or self.event is not None and self.event[0] == 27
    
    def requeue(self):
        self.renderer._queue_render(self.world)

    @profile
    def _cycle(self):
        self.requeue()
        self.draw_world()
        super(Window, self)._cycle()
        self.event = self._get_event()
        

class WindowWithHeadsUp(Window):
    def screen_init(self):
        super(WindowWithHeadsUp, self).screen_init()
        flags = pygame.NOFRAME

        self.workspace_ll = np.array([-25., -14.])

        win_res = (480, 270)
        self.workspace_size = 50, 28. #win_res
        self.size = np.array(win_res)
        self.screen = pygame.display.set_mode(win_res, flags)

import matplotlib.pyplot as plt
# import plotutil
from pylab import Circle

class MatplotlibWindow(object):
    background = (0, 0, 0, 1)
    def __init__(self, *args, **kwargs):
        print 'constructor for MatplotlibWindow'
        self.fig = plt.figure(figsize=(3,2))
        print 1
        axes = plotutil.subplots(1, 1, hold=True, aspect=1, left_offset=0.1)
        print 2
        self.ax = axes[0,0]
        print 3
        self.ax.set_xlim([-25, 25])
        self.ax.set_ylim([-14, 14])
        print 4

        self.model_patches = dict()
        print 5
        super(MatplotlibWindow, self).__init__(*args, **kwargs)
        print 6

        self.mpl_background = (1, 1, 1, 1) # self.background
        self.ax.set_axis_bgcolor(self.mpl_background)
        print 7

    def draw_world(self):
        print 'matplotlib window, draw world'  
        # TODO make sure cursor is on top
        for model in self.model_patches:
            if model not in self.world.models:
                patch = self.model_patches[model]
                patch.set_facecolor(self.mpl_background[:3])
                patch.set_edgecolor(self.mpl_background[:3])

        for model in self.world.models:
            if isinstance(model, Sphere):
                if model not in self.model_patches:
                    self.model_patches[model] = Circle(np.zeros(2), radius=model.radius, color='white', alpha=0.5)
                    self.ax.add_patch(self.model_patches[model])

                patch = self.model_patches[model]
                patch.set_facecolor(model.color[:3])
                patch.set_edgecolor(model.color[:3])

                pos = model.xfm.move[[0,2]]
                patch.center = pos

        plt.draw()
        super(MatplotlibWindow, self).draw_world()

class Simple2DWindow(object):
    background = (1,1,1,1)
    def __init__(self, *args, **kwargs):
        self.models = []
        self.world = None
        self.event = None        
        super(Simple2DWindow, self).__init__(*args, **kwargs)

    def screen_init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()
        self.clock = pygame.time.Clock()

        flags = pygame.NOFRAME

        
        if config.recording_sys['make'] == 'plexon':
            self.workspace_bottom_left = (-18., -12.)
            self.workspace_top_right   = (18., 12.)
            win_res = (1000, 560)
        elif config.recording_sys['make'] == 'blackrock':
            border = 10.  # TODO -- difference between this and self.display_border?
            self.workspace_bottom_left = np.array([ 0. - border, 
                                                    0. - border])
            self.workspace_top_right   = np.array([85. + border, 
                                                   95. + border])
            win_res = (600, 600)
        else:
            raise Exception('Unknown recording_system!')

        self.workspace_x_len = self.workspace_top_right[0] - self.workspace_bottom_left[0]
        self.workspace_y_len = self.workspace_top_right[1] - self.workspace_bottom_left[1]

        self.display_border = 10

        self.size = np.array(win_res, dtype=np.float64)
        self.screen = pygame.display.set_mode(win_res, flags)
        self.screen_background = pygame.Surface(self.screen.get_size()).convert()
        self.screen_background.fill(self.background)

        x1, y1 = self.workspace_top_right
        x0, y0 = self.workspace_bottom_left
        self.normalize = np.array(np.diag([1./(x1-x0), 1./(y1-y0), 1]))
        self.center_xform = np.array([[1., 0, -x0], 
                                      [0, 1., -y0],
                                      [0, 0, 1]])
        self.norm_to_screen = np.array(np.diag(np.hstack([self.size, 1])))

        # the y-coordinate in pixel space has to be swapped for some graphics convention reason
        self.flip_y_coord = np.array([[1, 0, 0],
                                      [0, -1, self.size[1]],
                                      [0, 0, 1]])

        self.pos_space_to_pixel_space = np.dot(self.flip_y_coord, np.dot(self.norm_to_screen, np.dot(self.normalize, self.center_xform)))

        self.world = GroupDispl2D(self.models)
        self.world.init()

        #initialize surfaces for translucent markers
        TRANSPARENT = (255,0,255)
        self.surf={}
        self.surf['0'] = pygame.Surface(self.screen.get_size())
        self.surf['0'].fill(TRANSPARENT)
        self.surf['0'].set_colorkey(TRANSPARENT)

        self.surf['1'] = pygame.Surface(self.screen.get_size())
        self.surf['1'].fill(TRANSPARENT)
        self.surf['1'].set_colorkey(TRANSPARENT)        

         #values of alpha: higher = less translucent
        self.surf['0'].set_alpha(170) #Cursor
        self.surf['1'].set_alpha(130) #Targets

        self.surf_background = pygame.Surface(self.surf['0'].get_size()).convert()
        self.surf_background.fill(TRANSPARENT)

    def pos2pix(self, kfpos):
        # re-specify the point in homogenous coordinates
        pt = np.hstack([kfpos, 1]).reshape(-1, 1)

        # perform the homogenous transformation
        pix_coords = np.dot(self.pos_space_to_pixel_space, pt)

        pix_pos = np.array(pix_coords[:2,0], dtype=int)
        return pix_pos

    @profile
    def draw_world(self):
        #Refreshes the screen with original background
        self.screen.blit(self.screen_background, (0, 0))
        self.surf['0'].blit(self.surf_background,(0,0))
        self.surf['1'].blit(self.surf_background,(0,0))
        
        i = 0
        for model in self.world.models: #added 12-17-13 to make cursor appear on top of target
            if isinstance(model, Sphere):
                pos = model.xfm.move[[0,2]]
                pix_pos = self.pos2pix(pos)
                color = tuple(map(lambda x: int(255*x), model.color[0:3]))
                rad = model.radius
                pix_radius = self.pos2pix(np.array([model.radius, 0]))[0] - self.pos2pix([0,0])[0]

                #Draws cursor and targets on transparent surfaces
                pygame.draw.circle(self.surf[str(np.min([i,1]))], color, pix_pos, pix_radius)
                i += 1
            elif isinstance(model, Shape2D):
                if model.draw(self.surf[str(np.min([i,1]))], self.pos2pix):
                    i += 1
            else:
                pass

            # NOTE: code below is old and has been moved into the .draw()
            # method of the classes Circle, Sector, and Line in primitives.py

            # elif isinstance(model, Circle):
            #     TODO -- implement
            #     i += 1
            # elif isinstance(model, Sector):
            #     if model.visible:
            #         # center_pos = model.center_pos[[0,2]]
            #         color = tuple(map(lambda x: int(255*x), model.color[0:3]))
                    
            #         center_pos = model.center_pos
            #         start_angle = model.ang_range[0]
            #         stop_angle  = model.ang_range[1]
            #         radius = model.radius

            #         arc_angles = np.linspace(start_angle, stop_angle, 5)
            #         sector_pts = list(center_pos + radius*np.c_[np.cos(arc_angles), np.sin(arc_angles)])
            #         sector_pts.append(center_pos)
            #         point_list = [self.pos2pix(pt) for pt in sector_pts]
            #         pygame.draw.polygon(self.surf[str(np.min([i,1]))], color, point_list)
            #         i += 1
            # elif isinstance(model, Line):
            #     start_pos = model.start_pos
            #     angle = model.angle
            #     length = model.length
            #     width = model.width

            #     origin = np.zeros([0, 0])

            #     # plot line as thin rectangle
            #     pts = np.array([[0, width/2], 
            #                     [0, -width/2], 
            #                     [length, -width/2], 
            #                     [length, width/2]])

            #     # rotate rectangle to correct orientation
            #     rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            #     pts = np.dot(rot_mat, pts.T).T
            #     pts = pts + np.array([start_pos[0], start_pos[1]])

            #     color = tuple(map(lambda x: int(255*x), model.color[0:3]))

            #     point_list = [self.pos2pix(pt) for pt in pts]
            #     pygame.draw.polygon(self.surf[str(np.min([i,1]))], color, point_list, 0)
            #     i += 1
            # else:
            #     pass

        #Renders the new surfaces
        self.screen.blit(self.surf['0'], (0,0))
        self.screen.blit(self.surf['1'], (0,0))
        pygame.display.update()

    def requeue(self):
        '''
        Simulation 'requeue' does nothing because the simulation is lazy and
        inefficient and chooses to redraw the entire screen every loop
        '''
        
        pass

class WindowDispl2D(Simple2DWindow, Window):
    # TODO -- defining this __init__ is probably not necessary
    def __init__(self, *args, **kwargs):
        super(WindowDispl2D, self).__init__(*args, **kwargs)


class FakeWindow(object):
    '''
    A dummy class to secretly avoid rendering graphics without 
    the graphics-based tasks knowing about it. Used e.g. for simulation 
    purposes where the graphics only slow down the simulation.
    '''
    background = (1,1,1,1)
    def __init__(self, *args, **kwargs):
        self.models = []
        self.world = None
        self.event = None        
        super(FakeWindow, self).__init__(*args, **kwargs)

    def screen_init(self):
        pass

    def draw_world(self):
        pass

    def requeue(self):
        pass

    def _start_reward(self, *args, **kwargs):
        n_rewards = self.calc_state_occurrences('reward')
        if n_rewards % 10 == 0:
            print n_rewards
        super(FakeWindow, self)._start_reward(*args, **kwargs)

    def show_object(self, *args, **kwargs):
        pass

    def _get_event(self):
        pass


class FPScontrol(Window):
    '''A mixin that adds a WASD + Mouse controller to the window. 
    Use WASD to move in XY plane, q to go down, e to go up'''

    def init(self):
        super(FPScontrol, self).init()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.eyepos = [0,-self.screen_dist, 0]
        self.eyevec = [0,0]
        self.wasd = [False, False, False, False, False, False]

    def _get_event(self):
        retme = None
        for e in pygame.event.get([pygame.MOUSEMOTION, pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT]):
            moved = any(self.wasd)
            if e.type == pygame.MOUSEMOTION:
                self.world.xfm.rotate *= Quaternion.from_axisangle((1,0,0), np.radians(e.rel[1]*.1))
                self.world.xfm.rotate *= Quaternion.from_axisangle((0,0,1), np.radians(e.rel[0]*.1))
                self.world._recache_xfm()
            elif e.type == pygame.KEYDOWN:
                kn = pygame.key.name(e.key)
                if kn in ["escape", "q"]:
                    self.stop = True
                retme = (e.key, e.type)
            elif e.type == pygame.QUIT:
                self.stop = True

            if moved:
                self.set_eye(self.eyepos, self.eyevec, reset=True)
        return retme
