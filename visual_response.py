from riglib.experiment import consolerun
from tasks import Dots

if __name__ == "__main__":
	probabilities = [0.5, None]
    options = {
        "penalty_time": 5,
        "ignore_time": 4, 
        "rand_start": (1, 10), 
        "reward_time": 5, 
        "timeout_time": 3,
    }
    exp = consolerun(Dots, ("autostart","button","ignore_correctness"), probabilities, **options)
