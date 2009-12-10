from django.http import HttpResponse
import Image
import numpy as np

def barcode2d(request, idNum):
    response = HttpResponse(mimetype="image/png")
    idNum=int(idNum)
    bid = '%64s' %(bin(idNum).split('b')[1])
    print idNum, bid
    im = 255*np.array([c=='1' for c in bid]).reshape(4,-1)
    #print im
    Image.fromarray(im.astype('uint8')).save(response, 'PNG')
    return response
    
