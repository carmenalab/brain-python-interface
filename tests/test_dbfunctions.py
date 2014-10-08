#!/usr/bin/python
'''
A set of tests to ensure that all the dbfunctions are still functional
'''
from db import dbfunctions as dbfn
reload(dbfn)

id = 4150
te = dbfn.TaskEntry(id)

print dbfn.get_plx_file(id)
print dbfn.get_decoder_name(id)
print dbfn.get_decoder_name_full(id)
print dbfn.get_decoder(id)
print dbfn.get_params(id)
print dbfn.get_param(id, 'decoder')
print dbfn.get_date(id)
print dbfn.get_notes(id)
print dbfn.get_subject(id)
print dbfn.get_length(id)
print dbfn.get_success_rate(id)

id = 1956
print dbfn.get_bmiparams_file(id)

# TODO check blackrock file fns


