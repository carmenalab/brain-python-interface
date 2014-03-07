#!/bin/bash
# Script to configure the pythonpath for the BMI3D library
# To use, add the line 'source $CODE_PATH/pathconfig.sh' to your bashrc
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/db:$DIR/tests:$DIR/analysis
