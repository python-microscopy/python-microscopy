# Create your views here.
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from SampleDB.samples.models import *
from django.http import Http404
from django import forms
from datetime import datetime

def slide_detail(request, slideID):
    try:
        sl = Slide.objects.get(slideID=slideID)
    except Slide.DoesNotExist:
        raise Http404

    images = sl.images.order_by('timestamp')
    labels = sl.labelling.all()
    #print files

    return render_to_response('samples/slide_detail.html', {'slide':sl, 'images':images, 'labels':labels})

def slide_index(request):
    sl = Slide.objects.all()

    return render_to_response('samples/slide_list.html', {'slides':sl})

#class ImageFilterForm(forms.Form):


def image_list(request):
    filters = {}
    #print datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))

    if 'start_date' in request.REQUEST:
        #print datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))
        filters['timestamp__gte'] = datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))

    if 'end_date' in request.REQUEST:
        #print datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))
        filters['timestamp__lte'] = datetime(*([int(s) for s in request.REQUEST['end_date'].split('/')][::-1]))


    users = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('user_') and i[1] == '1']
    #print users
    if len(users) >  0:
        filters['userID__in'] = users

    imgs = Image.objects.filter(**filters)

    return render_to_response('samples/image_list.html', {'object_list':imgs})


def tag_hint(request):
    hints = TagName.objects.filter(name__startswith=request.REQUEST['tag'])
    #print request.REQUEST['tag'], hints

    return render_to_response('samples/autocomplete.html', {'hints':hints})

def tag_image(request, image_id):
    image_id = int(image_id)
    im = Image.objects.get(imageID=image_id)
    im.Tag(request.POST['tag'])

    return HttpResponseRedirect('/images/%d' % image_id)
    
