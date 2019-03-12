"""
Mock classes
"""
import shutil

class MockDatabase(object):
    """ mock for dbq module """
    def save_log(self, idx, log, dbname='default'):
        f = open(str(idx), "w")
        f.write(str(log))
        f.close()

    def save_data(self, filename, system, saveid, dbname="default"):
        shutil.copy(filename, str(saveid) + "." + system)