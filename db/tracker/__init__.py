import os
#os.nice(4)

try:
    import models
    for m in [models.Task, models.Feature, models.Generator, models.System]:
        m.populate()
except:
    import traceback
    traceback.print_exc()