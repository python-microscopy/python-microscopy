from django.shortcuts import render
from django.http import HttpResponse
from PYME.IO import clusterIO

# Create your views here.
def file(request, filename):
    print 'file'
    return HttpResponse(clusterIO.getFile(filename))

def listing(request, filename):
    #print 'listing'
    listing = clusterIO.listdir(filename)
    #print filename, listing
    dirs = []
    files = []
    for l in listing:
        if l.endswith('/'):
            dirs.append(l)
        else:
            files.append(l)

    dirs.sort()
    files.sort()

    path = filename.lstrip('/').rstrip('/').split('/')
    breadcrumbs = [{'dir': n, 'path': '/'.join(path[:(i+1)])} for i, n in enumerate(path)]

    context = {'dirname' : filename, 'files':files, 'dirs': dirs, 'breadcrumbs':breadcrumbs}
    return render(request, 'clusterbrowser/dirlisting.html', context)
    #return HttpResponse(clusterIO.listdir(filename))
