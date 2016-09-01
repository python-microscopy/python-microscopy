from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^list/(?P<filename>.*)$', views.listing, name='dirview'),
    url(r'^(?P<filename>.*)$', views.file, name='fileview'),
]