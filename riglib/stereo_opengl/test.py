if __name__ == "__main__":
    from window import Window
    from primitives import Cylinder, Plane
    win = Window()
    win.add_model(Cylinder(height=2, shader="flat", color=(0.6,0.4,0.4,1)).translate(1,2,-1))
    win.add_model(Cylinder(height=3, shader="default", color=(0.3, 0.3, 0.6,1)).translate(-1,1,-2))
    win.run()