import os

def initialize_db(rigname='bmi3d'):

	''' Takes the name of one of the rigs and sets the correct django settings file
	for the desired database. This function must be called BEFORE calling "from
	tracker import models". Once the import has already been called the database
	cannot be changed without starting a new python session.'''

	dbdict = {'bmi3d':'db.settings', 'exorig':'db.settings_exo'}
	try:
		dbname = dbdict[rigname]
	except:
		print "Unrecognized rig!"

	os.environ['DJANGO_SETTINGS_MODULE'] = dbname