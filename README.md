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
```bash
sudo xargs apt-get -y install < requirements.system
```

## Windows
Visual C++ Build tools (for the 'traits' package)

# Installation
```bash
git clone -b unstable_py3 https://github.com/carmenalab/brain-python-interface.git
cd brain-python-interface
pip3 install -r requirements.txt
pip3 install -e .
```

# set up the database
```bash
cd db
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py makemigrations tracker
python3 manage.py migrate                  # make sure to do this twice!
```

# start server
```bash
python3 manage.py runserver
```

# Setup
Once the server is running, open up Chrome and navigate to localhost:8000/setup
- Under "Subjects", make sure at least one subject is listed. A subject named "test" is recommended for separating exploration/debugging from real data. 
- Under "tasks", add a task to the system by giving it the python path for your task class. See documentation link above for details on how to write a task. There are a couple of built in tasks to help you get started. 

	- For example, you can add the built-in task `riglib.experiment.mocks.MockSequenceWithGenerator` just to check that the user interface works
	- If you want to try something graphical, you can add the built-in task `built_in_tasks.passivetasks.TargetCaptureVFB2DWindow`. This will be a "visual feedback" task in which a cursor automatically does the center-out task, a standard task in motor ephys. 


# Run a task
Navigate to http://localhost:8000/exp_log/ in chrome. Then press 'Start new experiment' and run your task. 

Papers which have used this package
-----------------------------------
- Ramos Murguialday et al., A Novel Implantable Hybrid Brain-Machine-Interface (BMI) for Motor Rehabilitation in Stroke Patients. IEEE NER 2019
- Khanna P. and Carmena J.M. (2017) Beta band oscillations in motor cortex reflect neural population signals that delay movement onset. eLife 6:e24573. doi:10.7554/eLife.24573.
- Moorman H.G.*, Gowda S.* and Carmena J.M. (2017) Control of redundant kinematic degrees of freedom in a closed-loop brain-machine interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering 25(6), pp. 750-760. doi:10.1109/TNSRE.2016.2593696.
- Shanechi M.M.*, Orsborn A.L.*, Moorman H.G.*, Gowda S.*, Dangi S., and Carmena J.M. (2017) Rapid control and feedback rates in the sensorimotor pathway enhance neuroprosthetic control. Nature Communications 8:13825. doi:10.1038/ncomms13825.
- Dangi S., Gowda S., Moorman H.G., Orsborn A.L., So K., Shanechi M. and Carmena J.M. (2014) Continuous closed-loop decoder adaptation with a recursive maximum likelihood algorithm allows for rapid performance acquisition in brain-machine interfaces. Neural Computation 12, pp. 1-29.
