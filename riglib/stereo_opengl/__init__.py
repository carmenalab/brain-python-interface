if __name__ == "__main__":
    from window import Test
    from primitives import Cylinder, Plane
    win = Test()
    win.add_model(Cylinder())
    win.start()
