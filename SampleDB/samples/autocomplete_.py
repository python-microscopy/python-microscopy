from django.http import HttpResponse
from SampleDB.samples.models import *
import json
#import Image
#import numpy as np

#def int2bin(n, count=24):
#    """returns the binary of integer n, using count number of digits"""
#    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def creator_autocomplete(request, cname):
    response = HttpResponse(mimetype="application/json")

    choices = list(set([e.creator for e in Slide.objects.filter(creator__startswith=cname)]))

    response.write(json.dumps(choices))

    return response

def slide_autocomplete(request, slref, cname):
    response = HttpResponse(mimetype="application/json")

    choices = list(set([e.reference for e in Slide.objects.filter(reference__startswith=slref, creator=cname)]))
    response.write(json.dumps(choices))

    return response

def structure_autocomplete(request, sname):
    response = HttpResponse(mimetype="application/json")

    choices = list(set([e.structure for e in Labelling.objects.filter(structure__startswith=sname)]))

    response.write(json.dumps(choices))

    return response