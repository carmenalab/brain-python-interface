brain-python-interface (a.k.a. bmi3d)
====================================
**This the unstable python 3 branch. It may not yet do what you want it to do.**

This package contains Python code to run electrophysiology behavioral tasks,
with emphasis on brain-machine interface (BMIs) tasks. This package differs 
from other similar packages (e.g., BCI2000--http://www.schalklab.org/research/bci2000) 
in that it is primarily intended for intracortical BMI experiments. 

This package has been used with the following recording systems:
- Omniplex neural recording system (Plexon, Inc.). 
- Blackrock NeuroPort

Code documentation can be found at http://carmenalab.github.io/bmi3d_docs/

Getting started
---------------
# Dependencies
## Linux/OS X
(none at this time)

## Windows
Visual C++ Build tools (for the 'traits' package)

# Installation
```bash
git clone https://github.com/carmenalab/brain-python-interface
cd brain-python-interface
pip install -r requirements.txt
pip install -e .
```

# set up the database
```bash
cd db
python manage.py makemigrations
python manage.py migrate
python manage.py makemigrations tracker
python manage.py migrate                  # make sure to do this twice!
```

# start server
```bash
python manage.py runserver
```

# Setup
Once the server is running, open up Chrome and navigate to localhost:8000/setup
- Under 'subjects', make sure at least one subject is listed. A 'test' subject should be added by default. 
- Under 'tasks', add a task to the system by giving it the python path for your task class. See documentation link above for details on how to write a task. For example, you can add the built-in task riglib.experiment.mocks.MockSequenceWithGenerator just to get up and running

# Run a task
Navigate to http://localhost:8000/exp_log/ in chrome. Then press 'Start new experiment' and run your task. 

Papers which have used this package
-----------------------------------
- Ramos Murguialday et al., A Novel Implantable Hybrid Brain-Machine-Interface (BMI) for Motor Rehabilitation in Stroke Patients. IEEE NER 2019
- Khanna P. and Carmena J.M. (2017) Beta band oscillations in motor cortex reflect neural population signals that delay movement onset. eLife 6:e24573. doi:10.7554/eLife.24573.
- Moorman H.G.*, Gowda S.* and Carmena J.M. (2017) Control of redundant kinematic degrees of freedom in a closed-loop brain-machine interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering 25(6), pp. 750-760. doi:10.1109/TNSRE.2016.2593696.
- Shanechi M.M.*, Orsborn A.L.*, Moorman H.G.*, Gowda S.*, Dangi S., and Carmena J.M. (2017) Rapid control and feedback rates in the sensorimotor pathway enhance neuroprosthetic control. Nature Communications 8:13825. doi:10.1038/ncomms13825.
- Dangi S., Gowda S., Moorman H.G., Orsborn A.L., So K., Shanechi M. and Carmena J.M. (2014) Continuous closed-loop decoder adaptation with a recursive maximum likelihood algorithm allows for rapid performance acquisition in brain-machine interfaces. Neural Computation 12, pp. 1-29.
