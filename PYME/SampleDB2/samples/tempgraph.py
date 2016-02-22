#!/usr/bin/python

###############
# tempgraph.py
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
from django.shortcuts import render_to_response
from samples.models import *
import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np
import time
from PYME.misc import tempDB
from datetime import datetime

def temprecord(request):
    numhours = 24
    if 'numhours' in request.REQUEST:
        numhours = int(request.REQUEST['numhours'])

    #print files

    return render_to_response('samples/temperature_record.html', {'numhours':numhours,})

def tempgraph(request, numhours):
    response = HttpResponse(mimetype="image/png")

    #nbins= 800


    #print start_date, end_date
    nhours = int(numhours)
    endtime= time.time()
    starttime = endtime - 3600*nhours

    times_1, temps_1 = tempDB.getEntries(starttime, endtime, 1)

    mask = (temps_1 < 0)
    mask[:-1] += (diff(times_1) > 20)

    temps_1 = np.ma.masked_array(temps_1, mask)

    times_2, temps_2 = tempDB.getEntries(starttime, endtime, 2)

    mask = (temps_2 < 0)
    mask[:-1] += (diff(times_2) > 20)

    temps_2 = np.ma.masked_array(temps_2, mask)

    times_3, temps_3 = tempDB.getEntries(starttime, endtime, 3)

    mask = (temps_3 < 0)
    mask[:-1] += (diff(times_3) > 20)

    temps_3 = np.ma.masked_array(temps_3, mask)

    dpi=100.
    f = figure(figsize=(1000/dpi, 600/dpi))
    #axes((.05, .05, .9, .9))


    plot(times_1, temps_1)
    plot(times_2, temps_2)
    plot(times_3, temps_3)
    xlabel('Time')
    ylabel('Temperature')
    legend(['Instrument Frame', 'AC control', 'Optical Table'], loc=2)
    title('Last %d hours' % nhours)
    
    #xlim(bins[0], bins[-1])

    xt = xticks()[0]

    #xt = linspace(dates[0], dates[-1], 5)
    #print xt

    if nhours < 24:
        dateform = '%H:%M'
    elif nhours < 72:
        dateform = '%H:%M %d/%m'
    else:
        dateform = '%d/%m'

    xticks(xt, [time.strftime(dateform, datetime.fromtimestamp(t).timetuple()) for t in xt])


    #box(False)

    f.savefig(response, dpi=dpi, format='png')
    return response
    
