from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^download/(?P<filename>.*)$', views.file, name='fileview'),
    url(r'^upload/(?P<directory>.*)$', views.upload, name='upload'),
    url(r'^mkdir/(?P<basedir>.*)$', views.mkdir, name='mkdir'),
    url(r'^_lite/$', views.listing_lite, name='lite'),
    url(r'^(?P<filename>.*)$', views.listing, name='dirview'),
]