from django.conf.urls import patterns, include, url
from samples.models import *
from django.conf import settings
from samples import autocomp_settings

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'SampleDB2.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    (r'^slides/$', 'samples.views.slide_index'),
    (r'^slides/(?P<object_id>.*)$', 'django.views.generic.list_detail.object_detail', {'queryset' : Slide.objects.all()}),
    (r'^images/$', 'samples.views.image_list'),
    (r'^images/(?P<image_id>.*)/tag$', 'samples.views.tag_image'),
    (r'^images/(?P<object_id>.*)$', 'django.views.generic.list_detail.object_detail', {'queryset' : Image.objects.all()}),
    

    (r'^thumbnails/(?P<filename>.*)$', 'PYME.dataBrowser.thumbnails.thumb'),
    (r'^open/(?P<filename>.*)$', 'PYME.dataBrowser.open.openFile'),

    (r'^barcode/(?P<idNum>.*)$', 'samples.barcode.barcode2d'),
    (r'^datehist/$', 'samples.dategraph.dategraph'),
    (r'^tempgraph/(?P<numhours>.*)$', 'samples.tempgraph.tempgraph'),
    (r'^temperature/$', 'samples.tempgraph.temprecord'),

    (r'^hint/tag$', 'samples.views.tag_hint'),

    (r'^booking/$', 'samples.views.booking'),
    (r'^queues/$', 'samples.analysisTasks.analysisTasks'),

    (r'^media/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.MEDIA_ROOT }),
    (r'^$','samples.views.default'),

    url('^autocomplete/', include(autocomp_settings.autocomplete.urls)),

    url(r'^admin/', include(admin.site.urls)),
)
