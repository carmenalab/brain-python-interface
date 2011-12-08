import os

import tornado.ioloop
import tornado.web

from ..experiment import featurelist
from ..tasks import tasklist

cwd = os.path.split(os.path.abspath(__file__))[0]

class CreateRecord(tornado.web.RequestHandler):
    def get(self):
        alltraits = []
        for name, task in tasklist.items():
            tlist = []
            traits = task.class_traits()
            for trait in task.class_editable_traits():
                if isinstance(traits[trait], tuple):
                    tlist.append((trait, len(traits[trait].default), traits[trait].desc))
                else:
                    tlist.append((trait, 1, traits[trait].desc))
            alltraits.append(tlist)

    	print alltraits
        self.render("resources/start.html", features=featurelist, tasks=tasklist, alltraits=alltraits)

class AJAXHandler(tornado.web.RequestHandler):
    def get(self, path):
    	pass


application = tornado.web.Application([
    (r"/create/", CreateRecord),
    (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(cwd, "resources")}),
    (r"/ajax/(.*)", AJAXHandler),
])
application.template_path = os.path.join(cwd, "resources" )

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()