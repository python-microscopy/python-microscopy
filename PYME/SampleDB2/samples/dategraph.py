#!/usr/bin/python

###############
# dategraph.py
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
from django.http import HttpResponse
from samples.models import *
import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np
import time
from datetime import datetime

def dategraph(request):
    response = HttpResponse(mimetype="image/png")
    filters = {}

    if 'start_date' in request.REQUEST:
        if not request.REQUEST['start_date'] == 'none':
            filters['timestamp__gte'] = datetime(*([int(s) for s in request.REQUEST['start_date'].split('/')][::-1]))

    if 'end_date' in request.REQUEST:
        if not request.REQUEST['start_date'] == 'none':
            filters['timestamp__lte'] = datetime(*([int(s) for s in request.REQUEST['end_date'].split('/')][::-1]))

    usernames = set([i.userID for i in Image.objects.all()])
    usernames = [u for u in usernames if (u.find('-') == -1) and (u.find(' ') ==-1)]

    users = [i[0].split('_')[1] for i in request.REQUEST.items() if i[0].startswith('user_') and i[1] == '1']
    #print users
    if len(users) >  0:
        filters['userID__in'] = users

    tagnames = [t.name for t in TagName.objects.all()]

    tags = [i[0].split('__')[1] for i in request.REQUEST.items() if i[0].startswith('tag__') and i[1] == '1']
    #print users
#    if len(tags) >  0:
#        filters['userID__in'] = users

    #tag_info = [userInfo(t, t in tags) for t in tagnames]

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

    imgs = Image.objects.filter(**filters).order_by('timestamp')

    #print len(imgs)
    #print imgs[4]
    start_date = imgs[0].timestamp
    end_date = imgs[len(imgs)-1].timestamp

    nbins = min((end_date - start_date).days+2, 100)


    #print start_date, end_date

    dpi=100.
    f = figure(figsize=(1000/dpi, 70/dpi))
    axes((.05, .3, .9, .7))


    dates = [time.mktime(im.timestamp.timetuple()) for im in imgs]
    #print dates
    #print nbins, dates[-1]
    N, bins = histogram(dates, linspace(dates[0], dates[-1], nbins))

    bar(bins[:-1], N,bins[1] - bins[0])
    yticks([])
    xlim(bins[0], bins[-1])

    xt = linspace(dates[0], dates[-1], 5)
    #print xt

    if (end_date - start_date).days > 60:
        dateform = '%b %y'
    else:
        dateform = '%d %b %y'

    xticks(xt, [time.strftime(dateform, datetime.fromtimestamp(t).timetuple()) for t in xt])


    box(False)

    f.savefig(response, dpi=dpi, format='png')
    return response
    
