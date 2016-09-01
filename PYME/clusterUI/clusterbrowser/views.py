from django.shortcuts import render
from django.http import HttpResponse
from PYME.IO import clusterIO

# Create your views here.
def file(request, filename, type='raw'):
    #print 'file'
    if type == 'raw':
        return HttpResponse(clusterIO.getFile(filename), content_type='')

def listing(request, filename):
    #print 'listing'
    if not filename.endswith('/'):
        filename = filename + '/'

    if filename == '/':
        filename = ''

    listing = clusterIO.listdir(filename)
    #print filename, listing
    dirs = []
    series = []
    files = []
    for l in listing:
        if l.endswith('/'):
            dirListing = clusterIO.listdir(filename + l)
            nFiles = len(dirListing)
            if not 'metadata.json' in dirListing:
                dirs.append({'name':l, 'numFiles' : nFiles})
            else:
                complete = 'events.json' in dirListing
                nFrames = len([1 for f in dirListing if f.endswith('.pzf')])

                series.append({'name':l, 'numFrames' : nFrames, 'complete':complete})
        else:
            files.append(l)

    dirs.sort()
    files.sort()

    path = filename.lstrip('/').rstrip('/').split('/')
    breadcrumbs = [{'dir': n, 'path': '/'.join(path[:(i+1)])} for i, n in enumerate(path)]



    context = {'dirname' : filename, 'files':files, 'dirs': dirs, 'series': series, 'breadcrumbs':breadcrumbs}
    return render(request, 'clusterbrowser/dirlisting.html', context)
    #return HttpResponse(clusterIO.listdir(filename))
