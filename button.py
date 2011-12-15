from riglib.experiment import consolerun
from tasks.button import ButtonTask

if __name__ == "__main__":
	probs = [0, 0, 1]
	#use [0, 1, 0] for right side, [1, 0, 0] for left side, [0, 0, 1] for either side
	exp = consolerun(ButtonTask, ("button",), probs=probs)