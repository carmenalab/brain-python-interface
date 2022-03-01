from .tracker.models import DataFile, System, TaskEntry
import os
import sys

# sample_blackrock_files = [r'/storage/blackrock/test.nev', r'/storage/blackrock/test.ns5']
sample_blackrock_files = [r'/storage/blackrock/sampledata001.nev', r'/storage/blackrock/sampledata001.ns5', r'/storage/blackrock/sample_ecog.ns3']
# sample_blackrock_files = [r'/storage/blackrock/sample_ecog.ns3']

entry_num = sys.argv[1]
entry = TaskEntry.objects.get(id=entry_num)
system = System.objects.get(name='blackrock')

# get hdf associated with this model
datafile_list = DataFile.objects.filter(entry=entry).filter(system=System.objects.get(name='hdf'))
assert(len(datafile_list) == 1)
hdf_file = datafile_list[0]
hdf_fname_wo_ext = os.path.splitext(os.path.split(hdf_file.path)[1])[0]

for file_ in sample_blackrock_files:
	ext = os.path.splitext(os.path.split(file_)[1])[1]  # either .nev or .nsx (e.g., .ns5)
	path = hdf_fname_wo_ext + ext
	fullpath = os.path.join(r'/storage/blackrock', path)

	print(path)
	print(fullpath)
	if os.path.isfile(fullpath):
		raise Exception('File %s already exists!' % fullpath)
	else:
		print("Executing system command: sudo cp %s %s" % (file_, fullpath))
		print('')
		os.system("sudo cp %s %s" % (file_, fullpath))


		print('Saving DataFile object to database with following params:')
		print('entry:', entry)
		print('system:', system)
		print('path:', path)
		print('')
		DataFile(entry=entry, system=system, path=path).save()


