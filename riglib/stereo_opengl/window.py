'''
Graphical display classes. Experimental tasks involving graphical displays
inherit from these classes.
'''
import os

import numpy as np
from OpenGL.GL import *

from ..experiment import LogExperiment
from ..experiment import traits

from .render import stereo, render
from .models import Group
from .xfm import Quaternion
from .primitives import Sphere, Cube, Chain
from .environment import Box
from .primitives import Cylinder, Sphere, Cone
import socket

try:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
except ImportError:
    import warnings
    warnings.warn('riglib/stereo_opengl/window.py: not importing name pygame')

# for WindowDispl2D only
from .primitives import Shape2D

monitor_res = dict(
    test_monitor = (1680, 1050),
    monitor_2D = (2560, 1440),
    monitor_3D = (1920*2, 1080),
)

class Window(LogExperiment):
    '''
    Generic stereo window
    '''
    status = dict(draw=dict(stop=None))
    state = "draw"
    stop = False

    # XPS computer
    # window_size = (1920*2, 1080)
    window_size = traits.Tuple(monitor_res['monitor_2D'], desc='Window size, in pixels')
    background = traits.Tuple((0.,0.,0.,1.), desc="Background color (R,G,B,A)")
    fullscreen = traits.Bool(True, desc="Fullscreen window")

    #Screen parameters, all in centimeters -- adjust for monkey
    screen_dist = traits.Float(44.5+3, desc="Screen to eye distance (cm)")
    screen_half_height = traits.Float(10.75, desc="Screen half height (cm)")
    iod = traits.Float(2.5, desc="Intraocular distance (cm)")     # intraocular distance

    show_environment = traits.Int(0, desc="Show wireframe box around environment")

    hidden_traits = ['screen_dist', 'screen_half_height', 'iod', 'show_environment', 'background']

    def __init__(self, *args, **kwargs):
        self.display_start_pos = kwargs.pop('display_start_pos', "0,0")
        super(Window, self).__init__(*args, **kwargs)

        self.models = []
        self.world = None
        self.event = None

        # os.popen('sudo vbetool dpms on')
        self.fov = np.degrees(np.arctan(self.screen_half_height/self.screen_dist))*2
        self.screen_cm = [2 * self.screen_half_height * self.window_size[0]/self.window_size[1], 2 * self.screen_half_height]

        if self.show_environment:
            self.add_model(Box())

    def set_os_params(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = self.display_start_pos
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"

    def screen_init(self):
        self.set_os_params()
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL | pygame.NOFRAME
        if self.fullscreen:
            flags = flags | pygame.FULLSCREEN
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS,1)
            self.screen = pygame.display.set_mode(self.window_size, flags)
        except:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS,0)
            self.screen = pygame.display.set_mode(self.window_size, flags)

        glEnable(GL_BLEND)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.background)
        glClearDepth(1.0)
        glDepthMask(GL_TRUE)
        glEnable(GL_CULL_FACE) # temporary solution to alpha blending issue with spheres. just draw the front half of the sphere
        glCullFace(GL_FRONT) # actually it's the back half of the sphere we want since everything is mirrored

        self.renderer = self._get_renderer()

        #this effectively determines the modelview matrix
        self.world = Group(self.models)
        self.world.init()

        #up vector is always (0,0,1), why would I ever need to roll the camera?!
        self.set_eye((0, -self.screen_dist, 0), (0,0))

        pygame.mouse.set_visible(False)

    def _get_renderer(self):
        near = 1
        far = 1024
        return stereo.MirrorDisplay(self.window_size, self.fov, near, far, self.screen_dist, self.iod)

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

    def remove_model(self, model):
        '''remove the second model from the last'''
        self.world.remove(model)

    def show_object(self, obj, show=False):
        '''
        Show or hide an object. This function is an abstraction so that tasks don't need to know about attach/detach
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

    def _test_stop(self, ts):
        '''
        Stop the task if the escape key is pressed, or if the super _test_stop instructs a stop
        '''
        super_stop = super(Window, self)._test_stop(ts)
        from pygame import K_ESCAPE
        key_stop = self.event is not None and self.event[0] == K_ESCAPE
        if key_stop:
            print("Window closed. Stopping task")
        return super_stop or key_stop
    
    def update_report_stats(self):
        self.reportstats['FPS'] = round(self.clock.get_fps(),2)
        return super().update_report_stats()
    
    def requeue(self):
        self.renderer._queue_render(self.world)

    def _cycle(self):
        super(Window, self)._cycle() # is this order intentional? why not cycle first then draw the screen?
        self.event = self._get_event()
        self.requeue()
        self.draw_world()


class WindowWithExperimenterDisplay(Window):
    hostname = socket.gethostname()
    # This class has a hard-coded window size
    if hostname == 'lynx':
        _stereo_window_flip = True
        _stereo_main_flip_z = True
        window_size = (1280+480, 1024)
        window_size1 = (1280, 1024)
        window_size2 = (480, 270)
    else:
        _stereo_window_flip = False
        _stereo_main_flip_z = False
        window_size = (1920 + 480, 1080)
        window_size1 = (1920, 1080)
        window_size2 = (480, 270)

    def __init__(self, *args, **kwargs):
        super(WindowWithExperimenterDisplay, self).__init__(*args, **kwargs)
        # This class has a hard-coded window size
        # self.window_size = (1920 + 480, 1080)

    def set_os_params(self):
        # NOTE: in Ubuntu Unity, setting the SDL_VIDEO_WINDOW_POS seems to be largely ignored.
        # You can set which screen the window appears on if you have a dual display, but you cannot set the exact position
        # Instead, you have to hard-code a render start location in the compiz-config settings manager
        # http://askubuntu.com/questions/452995/how-to-adjust-window-placement-in-unity-ubuntu-14-04-based-on-overlapping-top-b
        os.environ['SDL_VIDEO_WINDOW_POS'] = self.display_start_pos
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment_with_mini"

    def _get_renderer(self):
        near = 1
        far = 1024
        return stereo.DualMultisizeDisplay(self.window_size1, self.window_size2, self.fov, near, far, self.screen_dist, self.iod, flip=self._stereo_window_flip,
            flip_main_z = self._stereo_main_flip_z)


TRANSPARENT = (106,0,42)

class WindowDispl2D():
    '''For testing only -- draws world on a 2D screen. May cause mild confusion -- transforms 
    incoming 3D coordinates (x,y,z) into 2D coordinates (x,y) by mapping z onto y'''

    def __init__(self, *args, **kwargs):
        self.is_pygame_display = True
        super().__init__(*args, **kwargs)

    def screen_init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = self.display_start_pos
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()
        self.clock = pygame.time.Clock()

        flags = pygame.NOFRAME | pygame.DOUBLEBUF | pygame.HWSURFACE

        if self.fullscreen:
            flags = flags | pygame.FULLSCREEN

        self.display_border = 10

        self.size = np.array(self.window_size, dtype=np.float64)
        self.screen = pygame.display.set_mode(self.window_size, flags)
        self.screen_background = pygame.Surface(self.screen.get_size()).convert()
        self.screen_background.fill(self.background)

        x_max, y_max = (self.screen_cm[0]/2, self.screen_cm[1]/2)
        x_min, y_min = (-self.screen_cm[0]/2, -self.screen_cm[1]/2)
        self.normalize = np.array(np.diag([1./(x_max-x_min), 1./(y_max-y_min), 1]))
        self.center_xform = np.array([[1., 0, -x_min], 
                                      [0, 1., -y_min],
                                      [0, 0, 1]])
        self.norm_to_screen = np.array(np.diag(np.hstack([self.size, 1])))

        # the y-coordinate in pixel space has to be swapped graphics convention reasons
        self.flip_y_coord = np.array([[1, 0, 0],
                                      [0, -1, self.size[1]],
                                      [0, 0, 1]])
        self.pos_space_to_pixel_space = np.dot(self.flip_y_coord, np.dot(self.norm_to_screen, np.dot(self.normalize, self.center_xform)))

        self.world = Group(self.models)
        # Dont 'init' self.world in this Window. Just allocates a bunch of OpenGL stuff which is not necessary (and may not work in some cases)
        # self.world.init()

        #initialize surfaces for translucent markers
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

        pygame.mouse.set_visible(False)

        self.i = 0

    def pos2pix(self, kfpos):
        # re-specify the point in homogenous coordinates
        pt = np.hstack([kfpos, 1]).reshape(-1, 1)

        # perform the homogenous transformation
        pix_coords = np.dot(self.pos_space_to_pixel_space, pt)

        pix_pos = np.array(pix_coords[:2,0], dtype=int)
        return pix_pos

    def get_surf(self):
        return self.surf[str(np.min([self.i,1]))]

    def draw_model(self, model):
        '''
        Draw a single Model on the current surface, or recurse if the model is a composite model (i.e., a Group)
        '''
        color = tuple([int(255*x) for x in model.color[0:3]])
        if isinstance(model, Sphere):
            pos = model._xfm.move[[0,2]]
            pix_pos = self.pos2pix(pos)

            rad = model.radius
            pix_radius = self.pos2pix(np.array([model.radius, 0]))[0] - self.pos2pix([0,0])[0]

            #Draws cursor and targets on transparent surfaces
            pygame.draw.circle(self.get_surf(), color, pix_pos, pix_radius)

        elif isinstance(model, Shape2D):
            # model.draw() returns True if the object was drawn
            #   (which happens if the object's .visible attr is True)
            if model.draw(self.get_surf(), self.pos2pix):
                pass
        elif isinstance(model, Cube):
            pos = model.xfm.move[[0,2]]
            side_len = model.side_len

            left = pos[0] - side_len/2
            right = pos[0] + side_len/2
            top = pos[1] + side_len/2
            bottom = pos[1] - side_len/2

            top_left = np.array([left, top])
            bottom_right = np.array([right, bottom])
            top_left_pix_pos = self.pos2pix(top_left)
            bottom_right_pix_pos = self.pos2pix(bottom_right)

            rect = pygame.Rect(top_left_pix_pos, bottom_right_pix_pos - top_left_pix_pos)
            color = tuple([int(255*x) for x in model.color[0:3]])

            pygame.draw.rect(self.get_surf(), color, rect)

        elif isinstance(model, (Cylinder, Cone)):
            vec_st = np.array([0., 0, 0, 1])
            vec_end = np.array([0., 0, model.height, 1])

            cyl_xform = model._xfm.to_mat()
            cyl_start = np.dot(cyl_xform, vec_st)
            cyl_end = np.dot(cyl_xform, vec_end)
            pix_radius = self.pos2pix(np.array([model.radius, 0]))[0] - self.pos2pix([0,0])[0]

            pygame.draw.line(self.get_surf(), color, self.pos2pix(cyl_start[[0,2]]), self.pos2pix(cyl_end[[0,2]]), pix_radius)

            # print cyl_start, cyl_end

        elif isinstance(model, Group):
            for mdl in model:
                self.draw_model(mdl)

    def _draw_other(self):
        pass

    def draw_world(self):
        #Refreshes the screen with original background
        self.screen.blit(self.screen_background, (0, 0))
        self.surf['0'].blit(self.surf_background,(0,0))
        self.surf['1'].blit(self.surf_background,(0,0))

        # surface index
        self.i = 0

        for model in self.world.models:
            self.draw_model(model)
            self.i += 1

        #Renders the new surfaces
        self.screen.blit(self.surf['0'], (0,0))
        self.screen.blit(self.surf['1'], (0,0))

        self._draw_other()
        pygame.display.update()

    def requeue(self):
        '''
        Simulation 'requeue' does nothing because the simulation is lazy and
        inefficient and chooses to redraw the entire screen every loop
        '''
        pass


class Window2D():
    '''
    Render the world in 2D without lighting
    '''
    def screen_init(self):
        super().screen_init()
    #     glBlendEquation(GL_MAX)
    #     glBlendFunc(GL_ONE, GL_ONE) # temporary solution to weird blending issue with spheres
        glCullFace(GL_BACK) # temporary solution #2 to alpha blending issue with spheres. just draw the front half of the sphere

    def _get_renderer(self):
        return render.Renderer2D(self.screen_cm)

class FakeWindow(Window):
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
        self.world = Group(self.models)
        # self.world.init()

    def draw_world(self):
        pass

    def requeue(self):
        pass

    def _start_reward(self, *args, **kwargs):
        n_rewards = self.calc_state_occurrences('reward')
        if n_rewards % 10 == 0:
            print(n_rewards)
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
