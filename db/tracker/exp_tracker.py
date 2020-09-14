""" Singleton for interactions between the browser and riglib """
from .tasktrack import Track

def get():
	return Track.get_instance()