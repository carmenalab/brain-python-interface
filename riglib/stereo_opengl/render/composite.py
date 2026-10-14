import numpy as np
from OpenGL.GL import *

from .fbo import FBOrender, FBO
from ..utils import orthographic


class CompositeOverlay(FBOrender):
    """
    Combined renderer:
    1) Render a 2D overlay into an offscreen FBO
    2) Render the main 3D scene to the window

    The FBO color texture is exposed as ``overlay_texture`` and can be attached to
    a ``TexPlane`` in the 3D scene.
    """

    def __init__(self, window_size, fov, near, far, 
                 root=None, overlay_size=None,
                 overlay_projection=None, overlay_modelview=None,
                 overlay_clear_color=(0., 0., 0., 0.), overlay_screen_cm=None,
                 overlay_near=1, overlay_far=1024, **kwargs):
        super().__init__(window_size, fov, near, far, **kwargs)

        w, h = window_size
        if overlay_size is None:
            overlay_size = (int(w), int(h))
        self.overlay_size = (int(overlay_size[0]), int(overlay_size[1]))

        if overlay_projection is None:
            if overlay_screen_cm is None:
                aspect = float(self.overlay_size[0]) / float(self.overlay_size[1])
                overlay_screen_cm = (2 * aspect, 2)
            overlay_projection = orthographic(
                overlay_screen_cm[0], overlay_screen_cm[1], overlay_near, overlay_far
            )

        self.overlay_projection = overlay_projection
        self.overlay_modelview = np.eye(4) if overlay_modelview is None else overlay_modelview
        self.overlay_clear_color = overlay_clear_color
        self.root = root

        texture_opts = {'anisotropic_filtering': 4}
        self.overlay_fbo = FBO(["color0", "depth"], size=self.overlay_size, texture_opts=texture_opts)

    @property
    def overlay_texture(self):
        return self.overlay_fbo['color0']

    def draw_overlay(self, overlay_root, overlay_shader=None,
                     overlay_apply_default=False, overlay_requeue=False,
                     overlay_p_matrix=None, overlay_modelview=None, **overlay_kwargs):
        if overlay_root is None:
            return

        original_viewport = glGetIntegerv(GL_VIEWPORT)
        original_framebuffer = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        original_clear_color = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        glViewport(0, 0, self.overlay_size[0], self.overlay_size[1])
        glBindFramebuffer(GL_FRAMEBUFFER, self.overlay_fbo.fbo)
        glClearColor(*self.overlay_clear_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        if overlay_p_matrix is None:
            overlay_p_matrix = self.overlay_projection
        if overlay_modelview is None:
            overlay_modelview = self.overlay_modelview
        if 'light_direction' not in overlay_kwargs:
            overlay_kwargs['light_direction'] = self.light_direction

        # Match Renderer2D by default: use the passthrough2d + none shader path.
        if overlay_shader is None:
            overlay_shader = 'ui'
            overlay_apply_default = True

        if overlay_apply_default and overlay_shader is not None:
            # Force every overlay model into a single 2D-style shader pass.
            self._queue_render(overlay_root, shader=overlay_shader)
            super().draw(
                overlay_root,
                shader=overlay_shader,
                apply_default=False,
                requeue=False,
                p_matrix=overlay_p_matrix,
                modelview=overlay_modelview,
                **overlay_kwargs
            )
        else:
            super().draw(
                overlay_root,
                shader=overlay_shader,
                apply_default=False,
                requeue=overlay_requeue,
                p_matrix=overlay_p_matrix,
                modelview=overlay_modelview,
                **overlay_kwargs
            )

        glBindFramebuffer(GL_FRAMEBUFFER, int(original_framebuffer))
        glViewport(*original_viewport)
        glClearColor(*original_clear_color)

    def draw(self, overlay_root, root=None, overlay_shader=None,
             overlay_apply_default=False, overlay_requeue=False,
             overlay_p_matrix=None, overlay_modelview=None, overlay_kwargs=None,
             **kwargs):
        if overlay_kwargs is None:
            overlay_kwargs = {}

        if root is None:
            root = self.root

        if overlay_root is not None:
            self.draw_overlay(
                overlay_root,
                overlay_shader=overlay_shader,
                overlay_apply_default=overlay_apply_default,
                overlay_requeue=overlay_requeue,
                overlay_p_matrix=overlay_p_matrix,
                overlay_modelview=overlay_modelview,
                **overlay_kwargs
            )

        super().draw(root, requeue=True, **kwargs)