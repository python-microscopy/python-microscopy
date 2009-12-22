# Create your views here.
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from SampleDB.samples.models import *
from django.http import Http404
from django import forms
from django.template import RequestContext
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


class userInfo(object):
    def __init__(self, name, disp):
        self.name = name
        self.disp=disp

def image_list(request):
    filters = {}
    startNum = 0
    numResults = 50
    #print datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))

    if 'start_date' in request.REQUEST:
        #print datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))
        if not request.REQUEST['start_date'] == 'none':
            filters['timestamp__gte'] = datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))

    if 'end_date' in request.REQUEST:
        #print datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))
        if not request.REQUEST['start_date'] == 'none':
            filters['timestamp__lte'] = datetime(*([int(s) for s in request.REQUEST['end_date'].split('/')][::-1]))

    if 'start_num' in request.REQUEST:
        startNum = int(request.REQUEST['start_num'])

    if 'num_results' in request.REQUEST:
        numResults = int(request.REQUEST['num_results'])

    usernames = set([i.userID for i in Image.objects.all()])
    usernames = [u for u in usernames if (u.find('-') == -1) and (u.find(' ') ==-1)]

    users = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('user_') and i[1] == '1']
    #print users
    if len(users) >  0:
        filters['userID__in'] = users

    user_info = [userInfo(u, u in users) for u in usernames]

#    tagnames = set()
#    for i in Image.objects.all():
#        for t in i.GetAllTags():
#            tagnames.add(t)
#
#    tagnames = list(tagnames)

    tagnames = [t.name for t in TagName.objects.all()]

    tags = [i[0].split('__')[1] for i in request.REQUEST.items() if i[0].startswith('tag__') and i[1] == '1']
    #print users
#    if len(tags) >  0:
#        filters['userID__in'] = users

    tag_info = [userInfo(t, t in tags) for t in tagnames]

    ImageIDs = []
    for t in tags:
        print t
        try:
            tn = TagName.objects.get(name=t)
            print tn
            ImageIDs += [i.image.imageID for i in ImageTag.objects.filter(tag=tn)]
            ImageIDs += [f.file.ImageID.imageID for f in FileTag.objects.filter(tag=tn)]
            for s in SlideTag.objects.filter(tag=tn):
                for i in s.slide.images.all():
                    ImageIDs += [i.image.imageID for i in ImageTag.objects.filter(tag=tn)]
        except:
            pass

    if len(tags) > 0:
        filters['imageID__in'] = ImageIDs

    #print len(ImageIDs)

    imgs = Image.objects.filter(**filters).order_by('timestamp')

    #if len(tags) > 0:
    #    imgs = [i for i in imgs if i.HasTags(tags)]


    start_date = imgs[0].timestamp
    end_date = imgs[len(imgs)-1].timestamp

    #print request.META['QUERY_STRING']

    totalResultsNum = len(imgs)
    numResults = min(totalResultsNum, startNum + numResults) - startNum
    imgs = imgs[startNum:(startNum + numResults)]

    return render_to_response('samples/image_list.html', {'object_list':imgs, 'user_info':user_info, 'tag_info':tag_info, 'startNum':startNum, 'endNum':(startNum + numResults), 'totalNum':totalResultsNum, 'start_date':start_date, 'end_date':end_date, 'query': request.META['QUERY_STRING']},context_instance=RequestContext(request))


def tag_hint(request):
    hints = TagName.objects.filter(name__startswith=request.REQUEST['tag'])
    #print request.REQUEST['tag'], hints

    return render_to_response('samples/autocomplete.html', {'hints':hints})

def tag_image(request, image_id):
    image_id = int(image_id)
    im = Image.objects.get(imageID=image_id)
    im.Tag(request.POST['tag'])

    return HttpResponseRedirect('/images/%d' % image_id)
    
