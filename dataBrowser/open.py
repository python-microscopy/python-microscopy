from django.http import HttpResponseRedirect
import os
import sys
import subprocess


def openFile(request, filename):
    filename = '/' + filename
    if os.path.exists(filename):
        #return HttpResponse("Thumbnail for %s." % filename)
        if filename.endswith('.h5') or filename.endswith('.kdf'):
            if sys.platform == 'win32':
                subprocess.Popen('..\\DSView\\dh5view.cmd %s' % filename, shell=True)
            else:
                subprocess.Popen('../DSView/dh5view.py %s' % filename, shell=True)

        elif filename.endswith('.h5r'):
            if sys.platform == 'win32':
                subprocess.Popen('..\\Analysis\\LMVis\\VisGUI.cmd %s' % filename, shell=True)
            else:
                subprocess.Popen('../Analysis/LMVis/VisGUI.py %s' % filename, shell=True)
    
    
    return HttpResponseRedirect('/browse' + '/'.join(filename.split('/')[:-1]))
