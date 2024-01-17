from django.urls import re_path
from . import views

app_name='localization'

urlpatterns = [
    re_path(r'^settings/$', views.settings, name='analysis_settings'),
    re_path(r'^settings/(?P<analysisModule>.+)/$', views.settings, name='analysis_settings'),
    re_path(r'^localize/$', views.localize, name='localize'),
]