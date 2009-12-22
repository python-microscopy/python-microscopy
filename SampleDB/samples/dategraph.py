from django.http import HttpResponse
from SampleDB.samples.models import *
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

    imgs = Image.objects.filter(**filters).order_by('timestamp')

    #print imgs[1]
    start_date = imgs[0].timestamp
    end_date = imgs[len(imgs)-1].timestamp

    nbins = min((end_date - start_date).days, 100)


    #print start_date, end_date

    dpi=100.
    f = figure(figsize=(1000/dpi, 70/dpi))
    axes((.05, .3, .9, .7))


    dates = [time.mktime(im.timestamp.timetuple()) for im in imgs]
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
    
