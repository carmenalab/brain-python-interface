if __name__ == "__main__":
    from window import Window
    from primitives import Cylinder, Plane
    win = Window()
    win.add_model(Cylinder(height=2).translate(0,2,-1))
    win.start()
    win.join