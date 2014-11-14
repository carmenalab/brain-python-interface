#!/bin/bash

MATLAB_LIB_DIR=/Applications/MATLAB_R2013b.app/bin/maci64
MATLAB_INCLUDE_DIR=/Applications/MATLAB_R2013b.app/extern/include

DEST_LIB_DIR=/Users/sdangi/code/CereLink/Matlab/lib/osx64
DEST_INCLUDE_DIR=/Users/sdangi/code/CereLink/Matlab/include

mkdir $DEST_LIB_DIR
cp $MATLAB_LIB_DIR/libmex.dylib $DEST_LIB_DIR
cp $MATLAB_LIB_DIR/libmx.dylib $DEST_LIB_DIR
cp $MATLAB_LIB_DIR/libmat.dylib $DEST_LIB_DIR

cp $MATLAB_INCLUDE_DIR/*.h $DEST_INCLUDE_DIR