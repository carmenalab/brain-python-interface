#!/usr/bin/env python
"""
Simple validation script to test that FBO creates textures with mipmapping
and anisotropic filtering when passed to Overlay2DTo3D renderer.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

def test_fbo_texture_opts():
    """Test that FBO properly passes texture_opts to Texture constructor."""
    from riglib.stereo_opengl.render.fbo import FBO
    
    # Test 1: FBO accepts texture_opts parameter
    texture_opts = {'mipmap': True, 'anisotropic_filtering': 4}
    try:
        fbo = FBO(["color0", "depth"], size=(640, 480), texture_opts=texture_opts)
        print("✓ FBO accepts texture_opts parameter")
    except TypeError as e:
        print(f"✗ FBO TypeError: {e}")
        return False
    
    # Test 2: Color texture was created
    try:
        color_tex = fbo['color0']
        print(f"✓ FBO color0 texture created: {color_tex}")
    except KeyError as e:
        print(f"✗ Color texture not found: {e}")
        return False
    
    # Test 3: Check that texture has mipmap option set
    if hasattr(color_tex, 'opts'):
        if color_tex.opts.get('mipmap'):
            print(f"✓ Texture has mipmap=True")
        else:
            print(f"✗ Texture mipmap not set")
            return False
        
        if color_tex.opts.get('anisotropic_filtering', 0) > 0:
            print(f"✓ Texture has anisotropic_filtering={color_tex.opts['anisotropic_filtering']}")
        else:
            print(f"✗ Texture anisotropic_filtering not set")
            return False
    
    return True

def test_overlay_renderer():
    """Test that Overlay2DTo3D creates FBO with texture filtering enabled."""
    from riglib.stereo_opengl.render.composite import Overlay2DTo3D
    from riglib.stereo_opengl.models import Group
    
    try:
        # Create a simple overlay root
        overlay_root = Group()
        
        # Initialize the overlay renderer with default settings
        renderer = Overlay2DTo3D(
            window_size=(640, 480),
            fov=60,
            near=0.1,
            far=100,
            overlay_root=overlay_root
        )
        
        # Test that overlay FBO was created with filtering
        color_tex = renderer.overlay_fbo['color0']
        
        if hasattr(color_tex, 'opts'):
            if color_tex.opts.get('mipmap'):
                print(f"✓ Overlay renderer FBO texture has mipmap=True")
            else:
                print(f"✗ Overlay renderer FBO texture mipmap not enabled")
                return False
            
            if color_tex.opts.get('anisotropic_filtering', 0) == 4:
                print(f"✓ Overlay renderer FBO texture has anisotropic_filtering=4")
            else:
                print(f"✗ Overlay renderer FBO texture anisotropic_filtering not set to 4")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Overlay renderer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing FBO texture filtering options...")
    print()
    
    # Run tests
    success = True
    
    print("Test 1: FBO texture_opts parameter")
    print("-" * 40)
    success = test_fbo_texture_opts() and success
    print()
    
    print("Test 2: Overlay2DTo3D renderer filtering")
    print("-" * 40)
    success = test_overlay_renderer() and success
    print()
    
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
