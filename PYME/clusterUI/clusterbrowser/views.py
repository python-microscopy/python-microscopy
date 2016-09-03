from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.views.decorators.csrf import csrf_exempt, csrf_protect

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

    if len(breadcrumbs) > 1:
        parent = breadcrumbs[-2]['path']
    elif len(breadcrumbs) == 1:
        parent = '/'
    else:
        parent = None


    context = {'dirname' : filename, 'files':files, 'dirs': dirs, 'series': series, 'breadcrumbs':breadcrumbs, 'parent' : parent}
    return render(request, 'clusterbrowser/dirlisting.html', context)
    #return HttpResponse(clusterIO.listdir(filename))

@csrf_exempt
def upload(request, directory):
    #use the temporary file handler by default as we want to be able to read files using PYME.IO.ImageStack
    #FIXME - this will not work for files with separate metadata
    request.upload_handlers.insert(0, TemporaryFileUploadHandler(request))
    return upload_files(request, directory)

@csrf_protect
def upload_files(request, directory):
    from PYME.IO import clusterIO

    files = request.FILES.getlist('file')
    for file in files:
        targetFilename = directory + file.name
        if clusterIO.exists(targetFilename):
            return HttpResponseForbidden('Upload failed [no files uploaded]. %s already exists on cluster' % targetFilename)

    for file in files:
        targetFilename = directory + file.name
        clusterIO.putFile(targetFilename, file.read())

    return HttpResponseRedirect(request.META['HTTP_REFERER'])

def mkdir(request, basedir):
    from PYME.IO import clusterIO
    newDirectory = request.POST.get('newDirectory', request.GET.get('newDirectory', None))

    if newDirectory is None or newDirectory == '':
        return HttpResponseForbidden('No directory name specified')

    newDirectory = (basedir + newDirectory).rstrip('/') + '/'

    if clusterIO.exists(newDirectory) or clusterIO.exists(newDirectory[:-1]):
        return HttpResponseForbidden('Directory already exists')

    clusterIO.putFile(newDirectory, '')

    return HttpResponse(newDirectory)
