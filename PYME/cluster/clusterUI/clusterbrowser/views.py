from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.views.decorators.csrf import csrf_exempt, csrf_protect, ensure_csrf_cookie

from wsgiref.util import FileWrapper

from PYME.IO import clusterIO
import os

# Create your views here.
def file(request, filename):
    type = request.GET.get('type', 'raw')
    #print 'file'
    if type == 'raw':
        return HttpResponse(clusterIO.get_file(filename, use_file_cache=False), content_type='')
    elif type in  ['tiff', 'h5']:
        from PYME.IO import image
        import tempfile
        img = image.ImageStack(filename='pyme-cluster://%s/%s' % (clusterIO.local_serverfilter, filename.rstrip('/')), haveGUI=False)

        if type == 'tiff':
            ext = '.tif'
        else:
            ext = '.' + type

        fn = os.path.splitext(os.path.split(filename.rstrip('/'))[-1])[0] + ext

        #note we are being a bit tricky here to ensure our temporary file gets deleted when we are done
        # 1) We create the temporary file using the tempfile module. This gets automagically deleted when we close the
        #    file (at the end of the with block)
        # 2) We pass the filename of the temporary file to img.Save. This will mean that a second file object / file handle
        #    gets created, the contents get written, and the file gets closed
        with tempfile.NamedTemporaryFile(mode='w+b', suffix=ext) as outf:
            img.Save(outf.name)

            #seek to update temporary file (so that it knows the new length)
            outf.seek(0)

            wrapper = FileWrapper(outf)
            response = HttpResponse(wrapper, content_type='image/%s' % ext.lstrip('.'))
            response['Content-Disposition'] = 'attachment; filename=%s' % fn
            response['Content-Length'] = os.path.getsize(outf.name)
            return response

def _get_listing(filename):
    from PYME.IO import clusterListing as cl
    #print 'listing'
    if not filename.endswith('/'):
        filename = filename + '/'

    if filename == '/':
        filename = ''

    listing = clusterIO.listdirectory(filename)
    #print filename, listing
    dirs = []
    series = []
    files = []
    
    filenames = sorted(listing.keys())
    for fn in filenames:
        file_info = listing[fn]
        if file_info.type & cl.FILETYPE_SERIES:
            complete = (file_info.type & cl.FILETYPE_SERIES_COMPLETE) > 0
            nFrames = file_info.size - 3 #assume we have metadata.json, events.json, and final_metadata.json - all others are  frames
            series.append({'name': fn, 'numFrames': nFrames, 'complete': complete,
                           'cluster_uri': (('pyme-cluster://%s/' % clusterIO.local_serverfilter) + filename + fn).rstrip('/')})

        elif file_info.type & cl.FILETYPE_DIRECTORY:
            dirs.append({'name': fn, 'numFiles': file_info.size})
        else:
            files.append(fn)

    #dirs.sort()
    #files.sort()

    path = filename.lstrip('/').rstrip('/').split('/')
    breadcrumbs = [{'dir': n, 'path': '/'.join(path[:(i + 1)])} for i, n in enumerate(path)]

    if len(breadcrumbs) > 1:
        parent = breadcrumbs[-2]['path']
    elif len(breadcrumbs) == 1:
        parent = '/'
    else:
        parent = None

    return {'dirname': filename, 'files': files, 'dirs': dirs, 'series': series, 'breadcrumbs': breadcrumbs,
               'parent': parent}


@ensure_csrf_cookie
def listing(request, filename):
    context = _get_listing(filename)
    return render(request, 'clusterbrowser/dirlisting.html', context)
    #return HttpResponse(clusterIO.listdir(filename))

def listing_lite(request):
    """A version of the listing to use in file selection boxes and the like"""
    filename = request.GET.get('path', '')
    context = _get_listing(filename)
    return render(request, 'clusterbrowser/lightlisting.html', context)




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
        clusterIO.put_file(targetFilename, file.read())

    return HttpResponseRedirect(request.META['HTTP_REFERER'])

def mkdir(request, basedir):
    from PYME.IO import clusterIO
    newDirectory = request.POST.get('newDirectory', request.GET.get('newDirectory', None))

    if newDirectory is None or newDirectory == '':
        return HttpResponseForbidden('No directory name specified')

    newDirectory = (basedir + newDirectory).rstrip('/') + '/'

    if clusterIO.exists(newDirectory) or clusterIO.exists(newDirectory[:-1]):
        return HttpResponseForbidden('Directory already exists')

    clusterIO.put_file(newDirectory, bytes('', 'utf-8'))

    return HttpResponse(newDirectory)
