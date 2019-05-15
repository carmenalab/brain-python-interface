#### ##################################################################### #####
#### ---- Starting point when only small local changes have been made ---- #####
#### ##################################################################### #####
FROM bmi3d:python
COPY bmi3d/ /code/bmi3d/
COPY bmi3d_tasks_analysis/ /code/bmi3d_tasks_analysis/

RUN python config/make_config.py --use-defaults && \
    python db/manage.py makemigrations && \
    python db/manage.py migrate

# Fix all .sh files that might have aquired windows line endings
RUN for f in $(find /code/ -name "*.sh"); \
    do echo "fixing: $f" && sed -i 's/\r$//' $f; \
    done  

CMD [ "/bin/bash", "./db/runserver.sh" ]