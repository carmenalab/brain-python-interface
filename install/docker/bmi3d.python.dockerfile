##### ################################################################### ####
##### ---- Starting point when python dependencies have been changed ---- ####
##### ################################################################### ####
FROM bmi3d:base

####### Set up directories and copy source code
RUN mkdir -v -p /code/src/
RUN mkdir -v -p /backup && chown root /backup
RUN mkdir -v -p /storage/plots && chown -R root /storage

# --- Expect cache invalidation here if source files have changed --- #
COPY bmi3d/ /code/bmi3d/
COPY bmi3d_tasks_analysis/ /code/bmi3d_tasks_analysis/

# Replace windows symlinks with unix ones in the new env
RUN rm /code/bmi3d/analysis && ln -s /code/bmi3d_tasks_analysis/analysis/ /code/bmi3d/analysis \
 && rm /code/bmi3d/tasks    && ln -s /code/bmi3d_tasks_analysis/tasks/    /code/bmi3d/tasks

# Fix all .sh files that might have aquired windows line endings
RUN for f in $(find ./code/ -name "*.sh"); \
	do echo "fixing: $f" && sed -i 's/\r$//' $f; \
	done  

WORKDIR /code/bmi3d/
RUN mkdir -v log


###### Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set env vars for future reference
ENV BMI3D="/code/bmi3d" \
	PYTHONPATH="${PYTHONPATH}:/code/bmi3d/:/code/bmi3d_tasks_analysis"

