'''
Performance calcuating functions
'''
from django.http import HttpResponse
# from tasks import performance
from analysis import performance
import os

html_plot_dir = '/static/storage/plots/'
def _render_plots(plot_fname_ls):
    html = ''
    for fname in plot_fname_ls:
        html += '\n<img src="%s" alt="Summary image not found. Did you generate it?">' % fname
    return html

def block_summary(request, idx):
    te = performance._get_te(int(idx))
    plot_fnames = [os.path.join(html_plot_dir, x) for x in te.get_plot_fnames()]
    print plot_fnames
    return HttpResponse(_render_plots(plot_fnames))

def bmi_perf_summary(request, idx):
    fname = os.path.join(html_plot_dir, '%s_bmi_summary.png' % idx)
    return HttpResponse(_render_plots([fname]))
