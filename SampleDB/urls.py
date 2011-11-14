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
     (r'^admin/(.*)', admin.site.root),
     (r'^$','SampleDB.samples.views.default'),

     url('^autocomplete/', include(autocomp_settings.autocomplete.urls)),

)

#urlpatterns += patterns('',
#            url(r'^%s(?P<path>.*)$' % settings.AUTOCOMPLETE_MEDIA_PREFIX[1:],
#            'django.views.static.serve',
#            {'document_root': os.path.join(autocomplete.__path__[0], 'media')})
