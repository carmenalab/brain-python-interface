from riglib.experiment import consolerun
from tasks import RDS, RDS_half

if __name__ == "__main__":
    options = {
        "penalty_time": 5,
        "ignore_time": 4, 
        "rand_start": (1, 10), 
        "reward_time": 5, 
        "timeout_time": 3,
    }
    exp = consolerun(RDS, ("autostart","button", "ignore_correctness"), probs=[0.5, None], **options)
