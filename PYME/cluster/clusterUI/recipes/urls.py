from django.urls import re_path
from . import views

app_name='recipes'

urlpatterns = [
    re_path(r'^simple_form/$', views.recipe_form, name='recipe'),
    re_path(r'^run/$', views.run, name='runrecipe'),
    re_path(r'^run_template/$', views.run_template, name='runtemplate'),
    re_path(r'^svg/$', views.view_svg, name='svg'),
    re_path(r'^standalone/$', views.recipe_standalone, name='bakeshop'),
    re_path(r'^editor/$', views.recipe_editor, name='editor'),
    re_path(r'^editrun/$', views.recipe_editrun, name='editrun'),
    re_path(r'^template/$', views.recipe_template, name='recipetemplate'),
    re_path(r'^find_inputs$', views.get_input_glob, name='find_inputs'),
    re_path(r'^extra_inputs/$', views.extra_inputs, name='extra_inputs'),
]