'''
__init__ script for stereo_opengl module
'''

try:
    from .window import Window
    from .render import stereo
    from .textures import Texture
except:
    pass