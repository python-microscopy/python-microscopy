from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.status, name='status'),
    url(r'^queues/$', views.queues, name='queues'),
    url(r'^load/$', views.load, name='load'),
]