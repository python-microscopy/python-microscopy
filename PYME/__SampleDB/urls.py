#!/usr/bin/python

###############
# urls.py
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
from django.conf.urls.defaults import *
from SampleDB.samples.models import *
from django.conf import settings
#import settings

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

#import autocomplete
#import os

from SampleDB.samples import autocomp_settings

urlpatterns = patterns('',
    # Example:
    #(r'^SampleDB/', include('SampleDB.foo.urls')),
    #(r'^data/$', 'SampleDB.samples.views.slide_index'),
    (r'^slides/$', 'SampleDB.samples.views.slide_index'),
    #(r'^slides/$', 'django.views.generic.list_detail.object_list', {'queryset' : Slide.objects.all()}),
    #(r'^slides/(?P<slideID>.*)$', 'SampleDB.samples.views.slide_detail'),
    (r'^slides/(?P<object_id>.*)$', 'django.views.generic.list_detail.object_detail', {'queryset' : Slide.objects.all()}),
    #(r'^images/$', 'django.views.generic.list_detail.object_list', {'queryset' : Image.objects.all()}),
    #(r'^images/(?P<imageID>.*)$', 'SampleDB.samples.views.slide_detail'),

    #(r'^images/$', 'django.views.generic.list_detail.object_list', {'queryset' : Image.objects.all()}),
    (r'^images/$', 'SampleDB.samples.views.image_list'),
    (r'^images/(?P<image_id>.*)/tag$', 'SampleDB.samples.views.tag_image'),
    (r'^images/(?P<object_id>.*)$', 'django.views.generic.list_detail.object_detail', {'queryset' : Image.objects.all()}),
    

    (r'^thumbnails/(?P<filename>.*)$', 'PYME.dataBrowser.thumbnails.thumb'),
    (r'^open/(?P<filename>.*)$', 'PYME.dataBrowser.open.openFile'),
    #(r'^thumbnails/(?P<filename>.*)$', 'PYME.dataBrowser.thumbnails.thumb'),
    (r'^barcode/(?P<idNum>.*)$', 'SampleDB.samples.barcode.barcode2d'),
    (r'^datehist/$', 'SampleDB.samples.dategraph.dategraph'),
    (r'^tempgraph/(?P<numhours>.*)$', 'SampleDB.samples.tempgraph.tempgraph'),
    (r'^temperature/$', 'SampleDB.samples.tempgraph.temprecord'),

    #(r'^autocomplete/slide$', 'SampleDB.samples.autocomplete.slide_autocomplete'),
    #(r'^autocomplete/creator$', 'SampleDB.samples.autocomplete.creator_autocomplete'),
    #(r'^autocomplete/structure$', 'SampleDB.samples.autocomplete.structure_autocomplete'),

    (r'^hint/tag$', 'SampleDB.samples.views.tag_hint'),

    (r'^booking/$', 'SampleDB.samples.views.booking'),
    (r'^queues/$', 'SampleDB.samples.analysisTasks.analysisTasks'),

    (r'^media/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.MEDIA_ROOT }),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
     url(r'^admin/', include(admin.site.urls)),
     (r'^$','SampleDB.samples.views.default'),

     url('^autocomplete/', include(autocomp_settings.autocomplete.urls)),

)

#urlpatterns += patterns('',
#            url(r'^%s(?P<path>.*)$' % settings.AUTOCOMPLETE_MEDIA_PREFIX[1:],
#            'django.views.static.serve',
#            {'document_root': os.path.join(autocomplete.__path__[0], 'media')})
