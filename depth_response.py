from riglib.experiment import consolerun
from tasks import Dots

if __name__ == "__main__":
    options = {
        "penalty_time": 5,
        "ignore_time": 2, 
        "rand_start": (1, 10), 
        "reward_time": 5, 
        "timeout_time": 3, 
        "trial_probs": [0.5, None]
    }
    exp = consolerun(Dots, ("autostart","button"), **options)