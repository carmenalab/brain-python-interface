
brain-python-interface (a.k.a. bmi3d)
====================================
**This fork is for private use in aolabs and is not intended to be a stable release**

This package contains Python code to run electrophysiology behavioral tasks,
with emphasis on brain-machine interface (BMIs) tasks. This package differs
from other similar packages (e.g., BCI2000--http://www.schalklab.org/research/bci2000)
in that it is primarily intended for intracortical BMI experiments.

This package has been used with the following recording systems:
- Omniplex neural recording system (Plexon, Inc.).
- Blackrock NeuroPort
- White-matter eCube

(Outdated) code documentation can be found at http://carmenalab.github.io/bmi3d_docs/

Getting started
---------------
# Dependencies
## Ubuntu
```bash
sudo xargs apt-get -y install < requirements.system
```
## Linux/OS X
(none at this time)
16.04 64bit

#python version
3.7.8

## Windows
Visual C++ Build tools (for the 'traits' package)

## Installation
```bash
pip install aolab-bmi3d
```

# Full Installation
```bash
git clone -b develop https://github.com/aolabNeuro/brain-python-interface.git
cd brain-python-interface
pip3 install -r requirements.txt
pip3 install -e .
./install-rabbitmq.sh
```

## Installation in Docker
- Set up docker on Ubuntu following these instructions: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
- In the `install` folder, execute `build_docker.sh` to build the image
- Also in the `install` folder, execute `run_docker.sh` to load the image. Annoyingly at this time, because the source directory is mounted in a way that is volatile, the bash shell will 'reinstall' the bmi3d package every time you load the image

Graphics generation from inside the image has only been tested in Ubuntu. To test, load the image using `run_docker.sh` and
```bash
cd /src/tests/unit_tests/
python3 test_built_in_vfb_task.py
```
If successful, you'll see the pygame window pop up looking like a poorly-made video game. If unsuccessful, you'll see the graphics in the terminal itself in ASCII art.


# Setting up a test database
```bash
cd db
touch db.sql 				   # Make empty file for database
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py makemigrations tracker
python3 manage.py migrate                  # make sure to do this twice!
```

# Use
Several use cases:

## Start the full BMI3D
```bash
./runserver.sh
```

## Start just the server
```bash
python3 manage.py runserver
```

## Use the database models in analysis
```python
from db import dbfunctions as db
te = db.get_task_entries()
```

# Setup paths and configurations
Once the server is running, open up Chrome and navigate to localhost:8000/setup
- Under "Subjects", make sure at least one subject is listed. A subject named "test" is recommended for separating exploration/debugging from real data.
- Under "tasks", add a task to the system by giving it the python path for your task class. See documentation link above for details on how to write a task. There are a couple of built in tasks to help you get started.

    - For example, you can add the built-in task `riglib.experiment.mocks.MockSequenceWithGenerator` just to check that the user interface works
	- If you want to try something graphical, you can add the built-in task `built_in_tasks.passivetasks.TargetCaptureVFB2DWindow`. This will be a "visual feedback" task in which a cursor automatically does the center-out task, a standard task in motor ephys.


# Run a task
Navigate to http://localhost:8000/exp_log/ in chrome. Then press 'Start new experiment' and run your task.

# Remote access
On Ubuntu, use the `ufw` package to open the port that Django is using
```bash
sudo ufw allow 8000/tcp
sudo ufw allow ssh
sudo ufw enable    # this is needed even if raspi-config enables SSH
```

Then start the server with
```bash
python3 manage.py runserver 0.0.0.0:8000
```

# Troubleshooting
This package has a lot of dependencies which makes installation somewhat brittle due to versions of different dependencies not getting along.

- Installation in a virtual environment (see `venv` in python3) or in a Docker container is recommended to try to isolate the package from version conflict issues.
- Run scripts in `tests/unit_tests/` to try to isolate which components may not be working correctly. Issues in `riglib` will be easier to fix than issues in the database.

## Running javascript tests
Javascript tests require the qunit library. To run the tests from the browser, open up `db/html/static/resources/js/qunit.html` in your browser of choice for running experiments.

If you are developing in javascript, you may want to run the tests from the command line. To install the qunit command-line setup, execute as root
```bash
cd db/html/static/resources/js
apt install npm
npm install -g qunit
npm install jquery
npm install jsdom
```
Then run `qunit` to run the tests in the `test` folder.


Papers which have used this package
-----------------------------------
- Ramos Murguialday et al., A Novel Implantable Hybrid Brain-Machine-Interface (BMI) for Motor Rehabilitation in Stroke Patients. IEEE NER 2019
- Khanna P. and Carmena J.M. (2017) Beta band oscillations in motor cortex reflect neural population signals that delay movement onset. eLife 6:e24573. doi:10.7554/eLife.24573.
- Moorman H.G.*, Gowda S.* and Carmena J.M. (2017) Control of redundant kinematic degrees of freedom in a closed-loop brain-machine interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering 25(6), pp. 750-760. doi:10.1109/TNSRE.2016.2593696.
- Shanechi M.M.*, Orsborn A.L.*, Moorman H.G.*, Gowda S.*, Dangi S., and Carmena J.M. (2017) Rapid control and feedback rates in the sensorimotor pathway enhance neuroprosthetic control. Nature Communications 8:13825. doi:10.1038/ncomms13825.
- Dangi S., Gowda S., Moorman H.G., Orsborn A.L., So K., Shanechi M. and Carmena J.M. (2014) Continuous closed-loop decoder adaptation with a recursive maximum likelihood algorithm allows for rapid performance acquisition in brain-machine interfaces. Neural Computation 12, pp. 1-29.
