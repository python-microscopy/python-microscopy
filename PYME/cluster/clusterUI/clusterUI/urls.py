"""clusterUI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  re_path(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  re_path(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  re_path(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include#, path
from django.urls import path, re_path
from django.contrib import admin
from django.views.generic.base import RedirectView

urlpatterns = [
    re_path(r'^admin/', admin.site.urls),
    re_path(r'^accounts/', include('django.contrib.auth.urls')),
    re_path(r'^registration/', include(('accounts.urls', 'accounts'), namespace='accounts')),
    re_path(r'^files/', include('clusterbrowser.urls')),
    re_path(r'^status/', include('clusterstatus.urls')),
    #re_path(r'^localization/', include('localization.urls')),
    path('localization/', include('localization.urls')),
    path('recipes/', include('recipes.urls')),
    re_path(r'^$', RedirectView.as_view(url='/files/')), #redirect the base view to files for now
]
