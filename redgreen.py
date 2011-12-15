from riglib.experiment import consolerun
from tasks import redgreen

if __name__ == "__main__":
    options = {
        "rand_start": (1, 10), 
        "delay_range": (0.5, 5),
        "dot_radius": 100,
        "reward_time": 5, 
        "timeout_time": 3,
        "penalty_time": 5,
    }
    exp = consolerun(redgreen.RedGreen, ("autostart","button"), redgreen.gencoords(), **options)
