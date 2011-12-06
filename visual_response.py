from riglib.experiment import consolerun
from riglib.tasks import Dots

if __name__ == "__main__":
    options = {
        "flat_proportion": 0, 
        "penalty_time": 5,
        "ignore_time": 4, 
        "rand_start": (1, 10), 
        "reward_time": 5, 
        "timeout_time": 3
    }
    exp = consolerun(Dots, ("autostart","button","ignore_correctness"), **options)
    print exp.event_log
    print exp.state_log