# Create your views here.
from django.shortcuts import render_to_response
from SampleDB.samples.models import *
from django.http import Http404

def slide_detail(request, slideID):
    try:
        sl = Slide.objects.get(slideID=slideID)
    except Slide.DoesNotExist:
        raise Http404

    images = sl.images.all()
    labels = sl.labelling.all()
    #print files

    return render_to_response('templates/slide_detail.html', {'slide':sl, 'images':images, 'labels':labels})

def slide_index(request):
    sl = Slide.objects.all()

    return render_to_response('templates/slide_index.html', {'slides':sl})
    
