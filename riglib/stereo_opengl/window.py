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
from riglib import loc_config
from primitives import Cylinder, Sphere, Cone

try:
    import pygame
except:
    import warnings
    warnings.warn('riglib/stereo_opengl/window.py: not importing name pygame')


class Window(LogExperiment):
    status = dict(draw=dict(stop=None))
    state = "draw"
    stop = False

    # window_size = (1280*2, 1024) # kinarm rig
    window_size = (3840, 1080) # exo rig
    background = (0,0,0,1)
    fps = 60

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

        if self.show_environment:
            self.add_model(Box())

    def screen_init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0' #loc_config.display_start_pos
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

    def _cycle(self):
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

        self.workspace_ll = np.array([-25., -14.])

        win_res = (480, 270)
        self.workspace_size = 50, 28. #win_res
        self.size = np.array(win_res)
        self.screen = pygame.display.set_mode(win_res, flags)
        self.screen_background = pygame.Surface(self.screen.get_size()).convert()
        self.screen_background.fill(self.background)

        self.pix_per_m = 10.4 #38.4 #self.size/self.workspace_size

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
        # rescale the cursor position to (0,1)
        norm_workspace_pos = (kfpos - self.workspace_ll)/self.workspace_size

        # multiply by the workspace size in pixels 
        pix_pos = self.size*norm_workspace_pos

        # flip y-coordinate
        pix_pos[1] = self.size[1] - pix_pos[1]

        # cast to integer
        pix_pos = np.array(pix_pos, dtype=int) 
        return pix_pos

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
                pix_radius = int(rad * self.pix_per_m)

                #Draws cursor and targets on transparent surfaces
                pygame.draw.circle(self.surf[str(np.min([i,1]))], color, pix_pos, pix_radius)
                i += 1
            else:
                pass

        #Renders the new surfaces
        self.screen.blit(self.surf['0'], (0,0))
        self.screen.blit(self.surf['1'], (0,0))
        pygame.display.update()
        #self.clock.tick(self.fps)

    def requeue(self):
        '''
        Simulation 'requeue' does nothing because the simulation is lazy and
        inefficient and chooses to redraw the entire screen every loop
        '''
        
        pass

class WindowDispl2D(Simple2DWindow, Window):
    def __init__(self, *args, **kwargs):
        super(WindowDispl2D, self).__init__(*args, **kwargs)



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
