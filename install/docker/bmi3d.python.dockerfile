##### ################################################################### ####
##### ---- Starting point when python dependencies have been changed ---- ####
##### ################################################################### ####
FROM bmi3d:base

# --- Expect cache invalidation here if source files have changed --- #
COPY bmi3d/requirements.txt /code/bmi3d/requirements.txt

WORKDIR /code/bmi3d/

###### Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set env vars for future reference
ENV BMI3D="/code/bmi3d" \
	PYTHONPATH="${PYTHONPATH}:/code/bmi3d/:/code/bmi3d_tasks_analysis"

