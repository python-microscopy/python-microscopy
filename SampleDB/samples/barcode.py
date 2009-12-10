from django.http import HttpResponse
import Image
import numpy as np

def int2bin(n, count=24):
    """returns the binary of integer n, using count number of digits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def barcode2d(request, idNum):
    response = HttpResponse(mimetype="image/png")
    idNum=int(idNum)
    #bid = '%64s' %(bin(idNum).split('b')[1])
    bid = int2bin(idNum, 64)
    #print idNum, bid
    im = 255*np.array([c=='1' for c in bid]).reshape(4,-1)
    #print im
    Image.fromarray(im.astype('uint8')).save(response, 'PNG')
    return response
    
