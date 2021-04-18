from PYME import config
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
import logging
from PYME.misc.computerName import GetComputerName
server_filter = config.get('dataserver-filter', GetComputerName())
logger = logging.getLogger(__name__)

# Create your views here.

def recipe_form(request):
    """stand in until we have a better recipe GUI"""
    return render(request, 'recipes/form_recipe.html', {'serverfilter' : server_filter})

def recipe_standalone(request):
    """This allows file selection with globs like bakeshop"""
    return render(request, 'recipes/recipe_standalone.html', {'serverfilter' : server_filter})

def recipe_editor(request):
    """User interface for editing recipes"""
    return render(request, 'recipes/recipe_editor.html', {'serverfilter' : server_filter})

def recipe_editrun(request):
    """Edit then run a recipe (effectively a composition of recipe_editor and recipe_standalone)"""
    return render(request, 'recipes/recipe_editrun.html', {'serverfilter' : server_filter})


def recipe_template(request):
    """This allows file selection with globs like bakeshop"""
    return render(request, 'recipes/recipe_template.html', {'serverfilter' : server_filter})

def get_input_glob(request):
    from PYME.IO import clusterIO
    
    filepaths = clusterIO.cglob(request.GET.get('glob').lstrip('/'))
    
    return render(request, 'recipes/input_list.html', {'filepaths' : filepaths,'serverfilter' : server_filter})

def run(request):
    from PYME import config
    from PYME.cluster.rules import RecipeRule

    recipe_url = request.POST.get('recipeURL')
    output_directory = 'pyme-cluster://%s/%s' % (server_filter, request.POST.get('recipeOutputPath').lstrip('/'))
    fileNames = request.POST.getlist('files', [])
    
    if recipe_url is not None:
        recipeURI = ('pyme-cluster://%s/' % server_filter) + recipe_url.lstrip('/')
        rule = RecipeRule(recipeURI=recipeURI, output_dir=output_directory, inputs={'input': fileNames})
    else:
        recipe_text = request.POST.get('recipe_text')
        rule = RecipeRule(recipe=recipe_text, output_dir=output_directory, inputs={'input': fileNames})
    
    rule.push()

    return HttpResponseRedirect('/status/queues/')

def run_template(request):
    from PYME import config
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection
    from PYME.cluster.rules import RecipeRule
        
    recipeURI = 'pyme-cluster://%s/%s' % (server_filter, request.POST.get('recipeURL').lstrip('/'))
    output_directory = 'pyme-cluster://%s/%s' % (server_filter, request.POST.get('recipeOutputPath').lstrip('/'))


    recipe_text = unifiedIO.read(recipeURI).decode('utf-8')
    recipe = ModuleCollection.fromYAML(recipe_text)
    

    # handle templated userfile inputs - these will be loaded by e.g. unifiedIO later
    for file_input in recipe.file_inputs:
        input_url = 'pyme-cluster://%s/%s' %(server_filter,  request.POST.get('%sURL' % file_input).lstrip('/'))
        recipe_text = recipe_text.replace('{'+file_input +'}', input_url)
    
    rule = RecipeRule(recipe=recipe_text, output_dir=output_directory, 
                      inputs={'input': request.POST.getlist('files', [])})
    rule.push()

    return HttpResponseRedirect('/status/queues/')


def view_svg(request):
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection
    from PYME.recipes import recipeLayout

    recipeURI = ('pyme-cluster://%s/' % server_filter) + request.GET.get('recipeURL').lstrip('/')

    recipe = ModuleCollection.fromYAML(unifiedIO.read(recipeURI))

    svg = recipeLayout.to_svg(recipe.dependancyGraph())

    return HttpResponse(svg, content_type='image/svg+xml')

def extra_inputs(request):
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection

    recipeURI = ('pyme-cluster://%s/' % server_filter) + request.GET.get('recipeURL').lstrip('/')

    recipe = ModuleCollection.fromYAML(unifiedIO.read(recipeURI))
    
    return render(request, 'recipes/extra_inputs.html', {'file_inputs': recipe.file_inputs, 'serverfilter' : server_filter})


