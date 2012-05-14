import time
from riglib import plexnet

conn = plexnet.Connection("10.0.0.13", 6000)
conn.connect(256)
conn.select_spikes()

data = []
d = conn.get_data()
t = time.time()
conn.start_data()
while time.time() - t < 10:
    data.append(d.next())
conn.stop_data()
conn.disconnect()

