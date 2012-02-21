if __name__ == "__main__":
    from window import Test
    from models import Builtins
    win = Test()
    win.add_model(Builtins("teapot", 1))
    win.start()