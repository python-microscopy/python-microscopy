#from django.conf.urls import url
from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.status, name='status'),
    re_path(r'^queues/$', views.queues, name='queues'),
    re_path(r'^load/$', views.load, name='load'),
]