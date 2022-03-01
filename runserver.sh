#!/bin/bash

# Set display
HOST=`hostname -s`
if [ "$HOST" = "pagaiisland2" ]; then
    export DISPLAY=':0.1'
elif [ "$HOST" = "peco" ]; then
    export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
    export LIBGL_ALWAYS_INDIRECT=0
elif [ "$HOST" = "pagaiisland-surface" ]; then
    export DISPLAY=localhost:0
    export LIBGL_ALWAYS_INDIRECT=0
    echo "success"
fi

# Find the BMI3D directory
FILE=$(realpath "$0")
BMI3D=$(dirname $FILE)
cd $BMI3D/db/

# Start logging
if [ -z "$1" ] # no arguments
then 
    echo "Turning on logging..."
    # Make the log directory if it doesn't already exist
    mkdir -p $BMI3D/log
    /bin/bash ./runserver.sh -log | tee -a $BMI3D/log/runserver_log
    exit 0
fi

# #Check /storage (exist )
# storage=$(python $BMI3D/config_files/check_storage.py 2>&1)
# if [ $storage == 'False' ]; then
#     echo "/storage does not exist --> if on Ismore, must mount"
#     exit 1
# fi

# Make sure that the server is not already running in a different program
if [ `ps aux | grep "manage.py runserver" | grep python | wc -l` -gt 0 ]; then 
    echo "ERROR: runserver seems to have already been executed by a different program!"
    read -p "Do you want to restart? [y/n] " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        kill -9 `ps -C "python manage.py runserver" -o pid --no-headers`
    fi
fi
    
# Mount the neural recording system, if a mount point is specified in the config file
# if [ `cat $BMI3D/config | grep mount_point | wc -l` -gt 0 ]; then
#     MOUNT_POINT=`cat $BMI3D/config | grep mount_point | tr -d '[:blank:]' | cut -d '=' -f 2`
#     if [[ -z `mount | grep $MOUNT_POINT` ]]; then
#         echo "Mounting neural recording system computer at $MOUNT_POINT"
#         sudo mount $MOUNT_POINT
#     else
#         echo "Neural recording system computer already mounted at $MOUNT_POINT"
#     fi
# fi

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

# Activate the relevant environment
if  test -f "$BMI3D/env/bin/activate"; then 
    source $BMI3D/env/bin/activate
else
    echo "No environment found."
fi

trap "exit" INT TERM ERR
trap "kill 0" EXIT

# Start python processes
cd $BMI3D
python manage.py runserver 0.0.0.0:8000 --noreload &
celery -A bmi3d.db.tracker worker -l INFO &
celery flower -A bmi3d.db.tracker --address=0.0.0.0 --port=5555 &

# Start servernode-control
if [ "$HOST" = "pagaiisland2" ]; then
    gnome-terminal -- ssh 10.155.207.19 sh ~/start-servernode.sh
    sleep 1
    gnome-terminal -- $BMI3D/riglib/ecube/servernode-control
fi

wait
