import os
import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, "..", ".."))

os.nice(4)

import models
for m in [models.Task, models.Feature, models.Generator]:
    m.populate()