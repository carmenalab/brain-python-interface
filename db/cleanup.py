from .tracker import models

def hide_empty_task_entries(hide=False):
	tes_empty_report = models.TaskEntry.objects.filter(report='')

	# filter out task entries which somehow have data files associated with them.
	tes_to_hide = []
	for te in tes_empty_report:
		assoc_dfs = models.DataFile.objects.filter(entry_id=te.id)
		if len(assoc_dfs) == 0:
			tes_to_hide.append(te)

	# filter out TaskEntry records which are already hidden
	tes_to_hide = [te for te in tes_to_hide if te.visible]

	if hide:
		print("hiding %d entries" % len(tes_to_hide))
		for te in tes_to_hide:
			te.visible = False
			te.save()

def calc_distribution_of_n_trials():
	n_trials = dict()
	visible_tes = models.TaskEntry.objects.filter(visible=True)
	for te in visible_tes:
		print(te.id)
		try:
			n_trials[te.id] = te.offline_report()['Total trials']
		except:
			pass
	return n_trials