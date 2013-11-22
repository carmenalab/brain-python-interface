#!/usr/bin/pyhon
from db import dbfunctions as dbfn
from tasks import performance
import matplotlib.pyplot as plt
import plotutil
import numpy as np
from db.tracker import models
from itertools import izip

reload(performance)
reload(plotutil)
reload(dbfn)

ppf_fixed_blocks_nov10 = dbfn.TaskEntrySet.construct_from_queryset(name='BMI perf Nov 10', filter_fns=[dbfn._assist_level_0], subj='C', date=(11,10), task__name__startswith='bmi')
ppf_fixed_blocks_nov11 = dbfn.TaskEntrySet.construct_from_queryset(name='BMI perf Nov 11', filter_fns=[dbfn._assist_level_0], subj='C', date=(11,11), task__name__startswith='bmi')
ppf_fixed_blocks_nov12 = dbfn.TaskEntrySet.construct_from_queryset(name='BMI perf Nov 12', filter_fns=[dbfn._assist_level_0, dbfn.min_trials(1)], subj='C', date=(11,12), task__name__startswith='bmi')

plt.close('all')
ppf_fixed_blocks_nov11.plot_perf_summary(plotattr=lambda te: 'tau = %g' % te.tau)
plt.savefig('/storage/plots/cart20131111_bmi_summary.png', bbox_inches='tight')

ppf_fixed_blocks_nov12.plot_perf_summary(plotattr=lambda te: 'tau = %g' % te.tau)
plt.savefig('/storage/plots/cart20131112_bmi_summary.png', bbox_inches='tight')

ppf_fixed_blocks_nov10.plot_perf_summary(plotattr=lambda te: 'tau = %g' % te.tau)
plt.savefig('/storage/plots/cart20131110_bmi_summary.png', bbox_inches='tight')
plt.show()
