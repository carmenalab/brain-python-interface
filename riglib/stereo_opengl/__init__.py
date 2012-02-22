if __name__ == "__main__":
    from window import Test
    from primitives import Cylinder
    win = Test()
    win.add_model(Cylinder())
    win.run()