.. _bmi

Brain-machine interface (BMI) code
==================================

The BMI code is split between the "tasks" module and the "bmi" module. The "bmi" module contains
the "low-level" components of the BMI, including the decoding algorithm, the assist methods, 
the adaptive filtering techinques for parameter fitting, etc. The "tasks" module integrates these
components to enable the BMI subject to perform a target-capture task.


..  automodule:: riglib.bmi.bmi
    :members: