from PYME import config
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from PYME.misc.computerName import GetComputerName
server_filter = config.get('dataserver-filter', GetComputerName())

# Create your views here.

def recipe_form(request):
    """stand in until we have a better recipe GUI"""
    return render(request, 'recipes/form_recipe.html', {'serverfilter' : server_filter})

def recipe_standalone(request):
    """This allows file selection with globs like bakeshop"""
    return render(request, 'recipes/recipe_standalone.html', {'serverfilter' : server_filter})

def recipe_template(request):
    """This allows file selection with globs like bakeshop"""
    return render(request, 'recipes/recipe_template.html', {'serverfilter' : server_filter})

def get_input_glob(request):
    from PYME.IO import clusterIO
    
    filepaths = clusterIO.cglob(request.GET.get('glob').lstrip('/'))
    
    return render(request, 'recipes/input_list.html', {'filepaths' : filepaths,'serverfilter' : server_filter})

def run(request):
    from PYME import config
    if config.get('PYMERuleserver-use', True):
        from PYME.cluster.HTTPRulePusher import RecipePusher
    else:
        from PYME.cluster.HTTPTaskPusher import RecipePusher
    recipeURI = ('pyme-cluster://%s/' % server_filter) + request.POST.get('recipeURL').lstrip('/')

    pusher = RecipePusher(recipeURI=recipeURI)


    fileNames = request.POST.getlist('files', [])
    pusher.fileTasksForInputs(input=fileNames)


    return HttpResponseRedirect('/status/queues/')

def run_template(request):
    from PYME import config
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection
    
    
    if config.get('PYMERuleserver-use', True):
        from PYME.cluster.HTTPRulePusher import RecipePusher
    else:
        from PYME.cluster.HTTPTaskPusher import RecipePusher
        
    recipeURI = 'pyme-cluster://%s/%s' % (server_filter, request.POST.get('recipeURL').lstrip('/'))
    output_directory = 'pyme-cluster://%s/%s' % (server_filter, request.POST.get('recipeOutputPath').lstrip('/'))


    recipe_text = unifiedIO.read(recipeURI).decode('utf-8')
    recipe = ModuleCollection.fromYAML(recipe_text)
    
    for file_input in recipe.file_inputs:
        input_url = 'pyme-cluster://%s/%s' %(server_filter,  request.POST.get('%sURL' % file_input).lstrip('/'))
        recipe_text = recipe_text.replace('{'+file_input +'}', input_url)

    pusher = RecipePusher(recipe=recipe_text, output_dir=output_directory)
    
    fileNames = request.POST.getlist('files', [])
    pusher.fileTasksForInputs(input=fileNames)

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


