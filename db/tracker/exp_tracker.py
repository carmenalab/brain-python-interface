""" Singleton for interactions between the browser and riglib """
from .tasktrack import Track
import sys
if sys.platform == "win32":
	use_websock = False
else:
	use_websock = True

exp_tracker = Track(use_websock=use_websock)

def get():
	return exp_tracker