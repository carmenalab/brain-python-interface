
from Phidgets.Devices.InterfaceKit import InterfaceKit
kit = InterfaceKit()
kit.openPhidget()
kit.waitForAttach(1000)
print(kit.isAttached())
