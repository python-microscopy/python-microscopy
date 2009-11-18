from django.conf.urls.defaults import *

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    # Example:
    # (r'^dataBrowser/', include('dataBrowser.foo.urls')),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # (r'^admin/(.*)', admin.site.root),
    (r'^thumbnails/(?P<filename>.*)$', 'dataBrowser.thumbnails.thumb'),
    (r'^browse/(?P<dirname>.*)$', 'dataBrowser.dirView.viewdir'),
    (r'^open/(?P<filename>.*)$', 'dataBrowser.open.openFile'),
)
