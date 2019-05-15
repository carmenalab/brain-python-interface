""" Singleton for interactions between the browser and riglib """

from .tasktrack import Track
exp_tracker = Track(use_websock=False)


def get():
	return exp_tracker
