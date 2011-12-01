from riglib.experiment import consolerun
from riglib.tasks import Dots

if __name__ == "__main__":
    import json
    options = json.load(open("options.json"))
    options['rand_start'] = tuple(options['rand_start'])
    consolerun(Dots, ("autostart","button","ignore_correctness"), **options)