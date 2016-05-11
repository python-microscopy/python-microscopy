from django.conf.urls import patterns, include, url
from samples.models import *
from django.conf import settings
#from django.views.generic.detail import DetailView
from samples.views import SlideDetailView, ImageDetailView
from samples import autocomp_settings

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'SampleDB2.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    (r'^slides/$', 'samples.views.slide_index'),
    #(r'^slides/(?P<object_id>.*)$', 'django.views.generic.list_detail.object_detail', {'queryset' : Slide.objects.all()}),
    #(r'^slides/(?P<object_id>.*)$', DetailView.as_view(), {'queryset' : Slide.objects.all()}),
    url(r'^slides/(?P<object_id>.*)$', SlideDetailView.as_view()),
    (r'^images/$', 'samples.views.image_list'),
    (r'^images/(?P<image_id>.*)/tag$', 'samples.views.tag_image'),
    #(r'^images/(?P<object_id>.*)$', 'django.views.generic.list_detail.object_detail', {'queryset' : Image.objects.all()}),
    #(r'^images/(?P<object_id>.*)$', DetailView.as_view(), {'queryset' : Image.objects.all()}),
    url(r'^images/(?P<object_id>.*)$', ImageDetailView.as_view()),
    

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

    (r'^api/num_matching_slides', 'samples.api.num_matching_slides'),
    (r'^api/get_slide_info', 'samples.api.get_slide_info'),
    (r'^api/get_creator_choices', 'samples.api.get_creator_choices'),
    (r'^api/get_slide_choices', 'samples.api.get_slide_choices'),
    (r'^api/get_structure_choices', 'samples.api.get_structure_choices'),
    (r'^api/get_dye_choices', 'samples.api.get_dye_choices'),



    url('^autocomplete/', include(autocomp_settings.autocomplete.urls)),

    url(r'^admin/', include(admin.site.urls)),
)
