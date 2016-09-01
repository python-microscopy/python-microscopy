from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^download/(?P<filename>.*)$', views.file, name='fileview'),
    url(r'^(?P<filename>.*)$', views.listing, name='dirview'),
]