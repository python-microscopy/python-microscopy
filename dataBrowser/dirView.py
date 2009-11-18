import os.path
from django.http import HttpResponse
from django.shortcuts import render_to_response
import os
import glob
from PYME.misc.dirSize import getDirectorySize
import numpy

MINSIZE=1
ONEGSIZE=2

def sortDirs(dirs):
    mangledDirnames =[]
    for d in dirs:
        dparts = d['name'].split('-')
        mangledDirnames.append('-'.join(s.zfill(2) for s in dparts[::-1]))

    dirI = numpy.argsort(mangledDirnames)

    return list(numpy.array(dirs)[dirI])

def findAnalysis(filename):
    if not filename.endswith('.h5'):
        return None
    else:
        fileparts = filename.split('/')

        cand = '/'.join(fileparts[:-2] + ['analysis',] + fileparts[-2:]) + 'r'

        if os.path.exists(cand):
            return cand
        else:
            return None

def viewdir(request, dirname):
    if dirname.endswith('/'):
        dirname = dirname[:-1]
    dirname1 = '/' + dirname
    if os.path.exists(dirname1):
        #return HttpResponse("Thumbnail for %s." % filename)
        children = glob.glob(dirname1 + '/*')

        files = [{'name':os.path.split(f)[1], 'size': '%.2f MB' % (os.path.getsize(f)/1024.**2), 'analysis':findAnalysis(f)} for f in children if os.path.isfile(f)]
        dirs = [{'name':os.path.split(f)[1], 'size': '%.2f GB' % (getDirectorySize(f)/1024.**3), 'sizef': MINSIZE + numpy.sqrt(ONEGSIZE*getDirectorySize(f)/1024.**3)} for f in children if os.path.isdir(f)]

        root_dirs = dirname.split('/')

        dirs = sortDirs(dirs)

        rdstub = ''
        rdirs = []
        for rd in root_dirs:
            if len(rd) > 0:
                rdstub = rdstub + '/' + rd
                rdirs.append({'name':rd, 'path':rdstub})


        return render_to_response('templates/dirview.html', {'dirname':dirname, 'files': files, 'dirs':dirs, 'rootdirs':rdirs})
    else:
        return HttpResponse("Directory %s does not exist." % dirname)
