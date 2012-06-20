import time
from riglib.plexon import plexnet

conn = plexnet.Connection("10.0.0.13", 6000)
conn.connect(256, waveforms=False, analog=True)
conn.select_continuous()

data = []
d = conn.get_data()
t = time.time()
conn.start_data()
while time.time() - t < 60:
    data.append(d.next())
conn.stop_data()
conn.disconnect()

