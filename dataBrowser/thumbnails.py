from django.http import HttpResponse
import os
import Image
from PYME.FileUtils import thumbnailDatabase

def thumb(request, filename):
    filename = '/' + filename
    if os.path.exists(filename):
        #return HttpResponse("Thumbnail for %s." % filename)
        
        im = thumbnailDatabase.getThumbnail(filename)
        if im == None:
            return HttpResponse("Could not generate thumbnail for %s." % filename)

        response = HttpResponse(mimetype="image/png")
        Image.fromarray(im).save(response, 'PNG')
        return response
    else:
        return HttpResponse("File %s does not exist." % filename)
