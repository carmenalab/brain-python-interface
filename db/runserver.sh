#!/bin/bash

# Set display to display 0
DISPLAY=`ps aux | grep -o "/usr/bin/X :[0-9]" | grep -o ":[0-9]"`
if [ $DISPLAY=='' ]; then
    DISPLAY=':0'
fi
#DISPLAY=':0'

# Make sure that the environment variable BMI3D is installed
if [ -z "$BMI3D" ]
    then
    echo "ERROR: Need to define the BMI3D environment variable!"
    exit 1
fi

#Check /storage (exist )
storage=$(python $BMI3D/config_files/check_storage.py 2>&1)
if [ $storage == 'False' ]; then
    echo "/storage does not exist --> if on Ismore, must mount"
    exit 1
fi

# Make sure that the server is not already running in a different program
if [ `ps aux | grep "manage.py runserver" | grep python | wc -l` -gt 0 ]; then 
    echo "ERROR: runserver seems to have already been executed by a different program!"
    exit 1
fi

# Check that a config file is in the correct place, $BMI3D/config
if [ ! -e $BMI3D/config_files/config ]; then 
    echo "ERROR: cannot find config file! Did you run $BMI3D/config_files/make_config.py?"
    exit 1
fi
    
# Mount the neural recording system, if a mount point is specified in the config file
if [ `cat $BMI3D/config | grep mount_point | wc -l` -gt 0 ]; then
    MOUNT_POINT=`cat $BMI3D/config | grep mount_point | tr -d '[:blank:]' | cut -d '=' -f 2`
    if [[ -z `mount | grep $MOUNT_POINT` ]]; then
        echo "Mounting neural recording system computer at $MOUNT_POINT"
        sudo mount $MOUNT_POINT
    else
        echo "Neural recording system computer already mounted at $MOUNT_POINT"
    fi
fi

# Print the date/time of the server (re)start
echo "Time at which runserver.sh was executed:"
date

# Print the most recent commit used at the time this script is executed
echo "Hash of HEAD commit at time of execution"
git --git-dir=$BMI3D/.git --work-tree=$BMI3D rev-parse --short HEAD

# Print the status of the BMI3D code so that there's a visible record of which files have changed since the last commti
echo "Working tree status at time of execution"
git --git-dir=$BMI3D/.git --work-tree=$BMI3D status

echo
echo
echo

##### all the previous stuff logging info sent to file
echo "Time at which runserver.sh was executed:" > $BMI3D/log/runserver_log
date >> $BMI3D/log/runserver_log 

# Print the most recent commit used at the time this script is executed
echo "Hash of HEAD commit at time of execution" >> $BMI3D/log/runserver_log  
git --git-dir=$BMI3D/.git --work-tree=$BMI3D rev-parse --short HEAD >> $BMI3D/log/runserver_log  

# Print the status of the BMI3D code so that there's a visible record of which  files have changed since the last commti
echo "Working tree status at time of execution" >> $BMI3D/log/runserver_log   
git --git-dir=$BMI3D/.git --work-tree=$BMI3D status >> $BMI3D/log/runserver_log   

    
trap ctrl_c INT SIGINT SIGKILL SIGHUP

MANAGER=$BMI3D/db/manage.py

# Start python processes and save their PIDs (stored in the bash '!' variable 
# immediately after the command is executed)
python $MANAGER runserver 0.0.0.0:8000 --noreload &
DJANGO=$!
python $MANAGER celery worker &
CELERY=$!
#python $MANAGER celery flower --address=0.0.0.0 &
#FLOWER=$!

# Define what happens when you hit control-C
function ctrl_c() {
	kill -9 $DJANGO
	kill $CELERY
	#kill $FLOWER
    kill -9 `ps aux | grep python | grep manage.py | tr -s " " | cut -d " " -f 2`
	# kill -9 `ps -C 'python manage.py' -o pid --no-headers`
}

# Run until the PID stored in $DJANGO is dead
wait $DJANGO
kill $CELERY
#kill $FLOWER
