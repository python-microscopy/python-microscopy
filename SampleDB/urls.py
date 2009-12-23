from django.conf.urls.defaults import *
from SampleDB.samples.models import *
from django.conf import settings
#import settings

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Example:
    #(r'^SampleDB/', include('SampleDB.foo.urls')),
    #(r'^data/$', 'SampleDB.samples.views.slide_index'),
    #(r'^slides/$', 'SampleDB.samples.views.slide_index'),
    (r'^slides/$', 'django.views.generic.list_detail.object_list', {'queryset' : Slide.objects.all()}),
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

    (r'^hint/tag$', 'SampleDB.samples.views.tag_hint'),

    (r'^media/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.MEDIA_ROOT }),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
     (r'^admin/(.*)', admin.site.root),
     (r'^$','SampleDB.samples.views.default'),

)
