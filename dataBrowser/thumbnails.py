from django.http import HttpResponse
import os
import Image
from PYME.FileUtils import thumbnailDatabase

def thumb(request, filename, size=200):
    filename = '/' + filename
    if os.path.exists(filename):
        #return HttpResponse("Thumbnail for %s." % filename)
        
        im = thumbnailDatabase.getThumbnail(filename)
        if im == None:
            return HttpResponse("Could not generate thumbnail for %s." % filename)

        response = HttpResponse(mimetype="image/png")

        xsize = im.shape[0]
        ysize = im.shape[1]

        zoom = float(size)/max(xsize, ysize)

        Image.fromarray(im).resize((int(zoom*ysize), int(zoom*xsize))).save(response, 'PNG')
        return response
    else:
        return HttpResponse("File %s does not exist." % filename)
