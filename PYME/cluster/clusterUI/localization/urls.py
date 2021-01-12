from django.conf.urls import url
from . import views

app_name='localization'

urlpatterns = [
    url(r'^settings/$', views.settings, name='analysis_settings'),
    url(r'^settings/(?P<analysisModule>.+)/$', views.settings, name='analysis_settings'),
    url(r'^localize/$', views.localize, name='localize'),
]