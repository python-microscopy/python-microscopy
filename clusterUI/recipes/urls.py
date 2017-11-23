from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^simple_form/$', views.recipe_form, name='recipe'),
    url(r'^run/$', views.run, name='runrecipe'),
    url(r'^svg/$', views.view_svg, name='svg'),
    url(r'^standalone/$', views.recipe_standalone, name='bakeshop'),
    url(r'^find_inputs/$', views.get_input_glob, name='find_inputs'),
]