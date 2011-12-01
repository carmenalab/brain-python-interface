import os
import json

cwd = os.path.split(os.path.abspath(__file__))[0]
optfile = os.path.join(cwd, "options.json")

def reset():
	default = dict(
		timeout_time = 4,
		penalty_time = 5,
		reward_time = 5,
		flat_proportion = 0.5,
		rand_start = (1, 10)
	)
	options.update(default)
	save()

def save():
	json.dump(options, open(optfile, "w"), sort_keys=True, indent=4)

if not os.path.exists(optfile):
	options = dict()
	reset()
else:
	options = json.load(open(optfile))