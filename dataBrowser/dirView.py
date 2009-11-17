import os.path
from django.http import HttpResponse
from django.shortcuts import render_to_response
import os
import glob

def viewdir(request, dirname):
    dirname1 = '/' + dirname
    if os.path.exists(dirname1):
        #return HttpResponse("Thumbnail for %s." % filename)
        children = glob.glob(dirname1 + '/*')

        files = [{'name':os.path.split(f)[1], 'size': '%.2f MB' % (os.path.getsize(f)/1024.**2)} for f in children if os.path.isfile(f)]
        dirs = [os.path.split(f)[1] for f in children if os.path.isdir(f)]

        return render_to_response('templates/dirview.html', {'dirname':dirname, 'files': files, 'dirs':dirs})
    else:
        return HttpResponse("Directory %s does not exist." % dirname)
