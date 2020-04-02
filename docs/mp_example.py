import multiprocessing as mp
import time

#### Version 1: single-threaded
def target_fn():
	time.sleep(2) # simulate thinkin
	print("TARGET FUNCTION: done computing answer")
	return "answer"

t_start = time.time()
print("Single-process version")
target_fn()
for k in range(30):
	print("fake event loop, index %d, time since start of loop: %g" % (k, time.time() - t_start))
	time.sleep(0.1)

print("\n\n\n\n\n\n")

#### Version 2: multi-threaded
t_start = time.time()
print("Multi-process version")
proc = mp.Process(target=target_fn)
proc.start()
for k in range(30):
	print("fake event loop, index %d, time since start of loop: %g" % (k, time.time() - t_start))
	time.sleep(0.1)


print("\n\n\n\n\n\n")

#### Version 3: multi-threaded, alternate implementation
class TargetClass(mp.Process):
	def run(self):
		target_fn()

t_start = time.time()
print("Multi-process version")
p = TargetClass()
p.start()
for k in range(30):
	print("fake event loop, index %d, time since start of loop: %g" % (k, time.time() - t_start))
	time.sleep(0.1)