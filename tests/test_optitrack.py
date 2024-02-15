from riglib import experiment
from features.optitrack_features import Optitrack
from riglib.optitrack_client import optitrack
from datetime import datetime
import time
import numpy as np
import os
import natnet

import unittest


class TestOptitrack(unittest.TestCase):

    def test_client(self):
        optitrack_ip = '10.155.204.10'
        client = natnet.Client.connect(server=optitrack_ip)
        sys = optitrack.System(client)
        response = client.set_version(3,1)
        print(response)
        client._send_command_and_wait("LiveMode")
        time.sleep(1)
        print(sys.get())


    def test_datasource(self):
        optitrack_ip = '10.155.204.10'
        client = natnet.Client.connect(server=optitrack_ip)
        from riglib import source
        motiondata = source.DataSource(optitrack.make(optitrack.System, client, "rigid body", 1))
        motiondata.start()
        time.sleep(0.5)

        data = motiondata.get()
        motiondata.stop()

        # print(data)




if __name__ == '__main__':
    unittest.main()