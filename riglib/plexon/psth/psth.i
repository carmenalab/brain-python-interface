%module psth
%{
#define SWIG_FILE_WITH_INIT
#include "psth.h"
%}
#define __attribute__(x) 
%include <pybuffer.i>
%include "numpy.i"

%init %{
import_array();
%}

%pybuffer_mutable_binary(char* bufchan, size_t clen);
%pybuffer_mutable_binary(char* bufspikes, size_t slen);
%apply (unsigned int* ARGOUT_ARRAY1, int DIM1) {(unsigned int* counts, int countlen)};

%include "psth.h"

%clear (int* counts, int countlen);