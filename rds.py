from riglib.experiment import consolerun
from tasks import RDS

if __name__ == "__main__":
    exp = consolerun(RDS, ("autostart","ignore_correctness"), timeout_time=30)