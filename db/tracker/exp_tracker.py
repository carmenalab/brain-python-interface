""" Singleton for interactions between the browser and riglib """
from .tasktrack import Track
exp_tracker = Track()

def get():
	return exp_tracker