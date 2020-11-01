from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^simple_form/$', views.recipe_form, name='recipe'),
    url(r'^run/$', views.run, name='runrecipe'),
    url(r'^run_template/$', views.run_template, name='runtemplate'),
    url(r'^svg/$', views.view_svg, name='svg'),
    url(r'^standalone/$', views.recipe_standalone, name='bakeshop'),
    url(r'^editor/$', views.recipe_editor, name='editor'),
    url(r'^editrun/$', views.recipe_editrun, name='editrun'),
    url(r'^template/$', views.recipe_template, name='recipetemplate'),
    url(r'^find_inputs/$', views.get_input_glob, name='find_inputs'),
    url(r'^extra_inputs/$', views.extra_inputs, name='extra_inputs'),
]