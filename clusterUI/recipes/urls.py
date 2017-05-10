from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^simple_form/$', views.recipe_form, name='recipe'),
    url(r'^run/$', views.run, name='runrecipe'),
    url(r'^svg/$', views.view_svg, name='svg'),
]