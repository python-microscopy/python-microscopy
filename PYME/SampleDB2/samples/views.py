#!/usr/bin/python

###############
# views.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#
################
# Create your views here.
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from samples.models import *
from django.http import Http404
from django import forms
from django.template import RequestContext
from datetime import datetime, timedelta

from django.views.generic.detail import DetailView

class SlideDetailView(DetailView):
    queryset = Slide.objects.all()
    
class ImageDetailView(DetailView):
    queryset = Image.objects.all()

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
    filters = {}

    creatornames = set([s.creator for s in Slide.objects.all()])
    creatornames = [u for u in creatornames if (u.find('-') == -1) and (u.find(' ') ==-1) and (u.find('.') ==-1)]

    creator = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('creator_') and i[1] == '1']
    #print users
    if len(creator) >  0:
        filters['creator__in'] = creator

    creator_info = [userInfo(u, u in creator) for u in creatornames]

    structures = set([l.structure for l in Labelling.objects.all()])

    structure = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('structure_') and i[1] == '1']

    if len(structure) >  0:
        filters['labelling__structure__in'] = structure

    structure_info = [userInfo(u, u in structure) for u in structures]

    slides = Slide.objects.filter(**filters)

    paginator = Paginator(slides, 25)

    page = request.REQUEST.get('page')
    #print page
    if page is None:
        page = '1'

    try:
        slp = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        slp = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        slp = paginator.page(paginator.num_pages)

    query = request.META['QUERY_STRING']
    query = '&'.join([q for q in query.split('&') if not q.startswith('page')])

    return render_to_response('samples/slide_list.html', {'slides':slp,'creator_info':creator_info,'structure_info':structure_info,'query':query,})

#class ImageFilterForm(forms.Form):


class userInfo(object):
    def __init__(self, name, disp):
        self.name = name
        self.disp=disp

def image_list(request):
    filters = {}
    startNum = 0
    numResults = 20
    min_h5r_size = 0
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

    if 'min_h5r_size' in request.REQUEST:
        min_h5r_size = int(request.REQUEST['min_h5r_size'])

    usernames = set([i.userID for i in Image.objects.all()])
    usernames = [u for u in usernames if (u.find('-') == -1) and (u.find(' ') ==-1)]

    users = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('user_') and i[1] == '1']
    #print users
    if len(users) >  0:
        filters['userID__in'] = users

    user_info = [userInfo(u, u in users) for u in usernames]

    structures = list(set([l.structure for l in Labelling.objects.all()]))
    structures.sort()

    structure = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('structure_') and i[1] == '1']

    if len(structure) >  0:
        filters['slideID__labelling__structure__in'] = structure

    structure_info = [userInfo(u, u in structure) for u in structures]

#    tagnames = set()
#    for i in Image.objects.all():
#        for t in i.GetAllTags():
#            tagnames.add(t)
#
#    tagnames = list(tagnames)

    tagnames = [t.name for t in TagName.objects.order_by('name')]

    tags = [i[0].split('__')[1] for i in request.REQUEST.items() if i[0].startswith('tag__') and i[1] == '1']
    #print users
#    if len(tags) >  0:
#        filters['userID__in'] = users

    tag_info = [userInfo(t, t in tags) for t in tagnames]

    ImageIDs = []
    for t in tags:
        #print t
        try:
            tn = TagName.objects.get(name=t)
            #print tn
            ImageIDs += [i.image.imageID for i in ImageTag.objects.filter(tag=tn)]
            #print [f.file.imageID for f in FileTag.objects.filter(tag=tn)]
            #print [f.file.imageID_id for f in FileTag.objects.filter(tag=tn)]
            ImageIDs += [f.file.imageID_id for f in FileTag.objects.filter(tag=tn)]
            for s in SlideTag.objects.filter(tag=tn):
                for i in s.slide.images.all():
                    ImageIDs += [i.image.imageID for i in ImageTag.objects.filter(tag=tn)]
        except:
            pass

    if len(tags) > 0:
        filters['imageID__in'] = ImageIDs

    #print len(ImageIDs)

    imgs = Image.objects.filter(**filters).order_by('timestamp')

    if min_h5r_size > 0:
        ImageIDs = [f.imageID_id for f in File.objects.filter(filename__endswith='h5r').filter(filesize__gt=1e6*min_h5r_size)]
        imgs = imgs.filter(imageID__in=ImageIDs)

    #imgs = imgs.filter(files__filesize__gt=10e6)

    #if len(tags) > 0:
    #    imgs = [i for i in imgs if i.HasTags(tags)]

    totalResultsNum = len(imgs)
    startNums = range(0,totalResultsNum, numResults)
    numResults = min(totalResultsNum, startNum + numResults) - startNum

    if not totalResultsNum == 0:
        start_date = imgs[0].timestamp
        end_date = imgs[len(imgs)-1].timestamp + timedelta(1)

        imgs = imgs[startNum:(startNum + numResults)]
    else:
        start_date = datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))
        end_date = datetime(*([int(s) for s in request.REQUEST['end_date'].split('/')][::-1]))

    if (startNum + numResults) < totalResultsNum:
        nextStartNum = startNum + numResults
    else:
        nextStartNum = 0


    query = request.META['QUERY_STRING']
    query = '&'.join([q for q in query.split('&') if not q.startswith('start_num')])

    return render_to_response('samples/image_list.html', {'object_list':imgs, 
                'user_info':user_info, 'tag_info':tag_info, 'structure_info': structure_info,
                'prevStartNum': max(0, startNum-numResults), 'nextStartNum':nextStartNum,
                'startNum':startNum, 'endNum':(startNum + numResults), 'totalNum':totalResultsNum,
                'startNums':startNums,
                'start_date':start_date, 'end_date':end_date, 'query': query, 'min_h5r_size' : min_h5r_size},
                context_instance=RequestContext(request))


def tag_hint(request):
    hints = TagName.objects.filter(name__startswith=request.REQUEST['tag'])
    #print request.REQUEST['tag'], hints

    return render_to_response('samples/autocomplete.html', {'hints':hints})

def tag_image(request, image_id):
    image_id = int(image_id)
    im = Image.objects.get(imageID=image_id)
    im.Tag(request.POST['tag'])

    return HttpResponseRedirect('/images/%d' % image_id)

def default(request):

    return render_to_response('samples/sample_main.html', {})

def booking(request):

    return render_to_response('samples/scope_booking.html', {})
    
