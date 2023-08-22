from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^download/(?P<filename>.*)$', views.file, name='fileview'),
    re_path(r'^upload/(?P<directory>.*)$', views.upload, name='upload'),
    re_path(r'^mkdir/(?P<basedir>.*)$', views.mkdir, name='mkdir'),
    re_path(r'^_lite/$', views.listing_lite, name='lite'),
    re_path(r'^(?P<filename>.*)$', views.listing, name='dirview'),
]