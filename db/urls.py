'''
Django's standard place to look for URL pattern matching. See 
https://docs.djangoproject.com/en/dev/topics/http/urls/
for more complete documentation
'''


#from django.conf.urls.defaults import *
from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

import trainbmi

urlpatterns = patterns('',
	(r'^$', 'tracker.views.list'),
    (r'^all/$', 'tracker.views.listall'),
    (r'^listdb/(?P<dbname>.+?)/ajax/exp_info/(?P<idx>\d+)/', 'tracker.ajax.exp_info'),    
    (r'^listdb/(?P<dbname>.+?)/ajax/task_info/(?P<idx>\d+)/', 'tracker.ajax.task_info'),    
    (r'^listdb/(?P<dbname>.+?)/$', 'tracker.views.listdb'),
    (r'^ajax/task_info/(?P<idx>\d+)/', "tracker.ajax.task_info"),
    (r'^ajax/exp_info/(?P<idx>\d+)/', 'tracker.ajax.exp_info'),
    (r'^ajax/hide_entry/(?P<idx>\d+)/', 'tracker.ajax.hide_entry'),    
    (r'^ajax/backup_entry/(?P<idx>\d+)/', 'tracker.ajax.backup_entry'),        
    (r'^ajax/gen_info/(?P<idx>\d+)/', 'tracker.ajax.gen_info'),
    (r'^ajax/save_notes/(?P<idx>\d+)/', 'tracker.ajax.save_notes'),
    (r'^all/ajax/task_info/(?P<idx>\d+)/', "tracker.ajax.task_info"),
    (r'^all/ajax/exp_info/(?P<idx>\d+)/', 'tracker.ajax.exp_info'),
    (r'^all/ajax/gen_info/(?P<idx>\d+)/', 'tracker.ajax.gen_info'),
    (r'^all/ajax/save_notes/(?P<idx>\d+)/', 'tracker.ajax.save_notes'),
    (r'^make_bmi/(?P<idx>\d+)/?', 'tracker.ajax.train_decoder_ajax_handler'), 
    (r'^ajax/setattr/(?P<attr>.+)/(?P<value>.+)', 'tracker.ajax.set_task_attr'),
    (r'^start/?', 'tracker.ajax.start_experiment'),
    (r'^next_exp/?', 'tracker.ajax.start_next_exp'),
    (r'^test/?', 'tracker.ajax.start_experiment', dict(save=False)),
    (r'^stop/?', 'tracker.ajax.stop_experiment'),
    (r'^enable_clda/?', 'tracker.ajax.enable_clda'),
    (r'^rewarddrain/(?P<onoff>\w+)/', 'tracker.ajax.reward_drain'),
    (r'^disable_clda/?', 'tracker.ajax.disable_clda'),
    (r'^sequence_for/(?P<idx>\d+)/', 'tracker.views.get_sequence'),
    (r'^RPC2/?', 'tracker.dbq.rpc_handler'),
    # Uncomment the admin/doc line below to enable admin documentation:
    (r'^admin/doc/', include('django.contrib.admindocs.urls')),
    # Uncomment the next line to enable the admin:
    (r'^admin/', include(admin.site.urls)),
)

