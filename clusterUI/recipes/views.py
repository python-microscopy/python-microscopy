from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

# Create your views here.

def recipe_form(request):
    """stand in until we have a better recipe GUI"""
    return render(request, 'recipes/form_recipe.html', {})

def recipe_standalone(request):
    """This allows file selection with globs like bakeshop"""
    return render(request, 'recipes/recipe_standalone.html', {})

def recipe_template(request):
    """This allows file selection with globs like bakeshop"""
    return render(request, 'recipes/recipe_template.html', {})

def get_input_glob(request):
    from PYME.IO import clusterIO
    
    filepaths = clusterIO.cglob(request.GET.get('glob').encode().lstrip('/'))
    
    return render(request, 'recipes/input_list.html', {'filepaths' : filepaths})

def run(request):
    from PYME import config
    if config.get('PYMERuleserver-use', True):
        from PYME.ParallelTasks.HTTPRulePusher import RecipePusher
    else:
        from PYME.ParallelTasks.HTTPTaskPusher import RecipePusher
    recipeURI = 'pyme-cluster:///' + request.POST.get('recipeURL').encode().lstrip('/')

    pusher = RecipePusher(recipeURI=recipeURI)


    fileNames = request.POST.getlist('files', [])
    pusher.fileTasksForInputs(input=fileNames)


    return HttpResponseRedirect('/status/queues/')

def run_template(request):
    from PYME import config
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection
    
    
    if config.get('PYMERuleserver-use', True):
        from PYME.ParallelTasks.HTTPRulePusher import RecipePusher
    else:
        from PYME.ParallelTasks.HTTPTaskPusher import RecipePusher
        
    recipeURI = 'pyme-cluster:///' + request.POST.get('recipeURL').encode().lstrip('/')


    recipe_text = unifiedIO.read(recipeURI)
    recipe = ModuleCollection.fromYAML(recipe_text)
    
    for file_input in recipe.file_inputs:
        input_url = 'pyme-cluster:///' + request.POST.get('%sURL' % file_input).encode().lstrip('/')
        recipe_text.replace('{'+file_input +'}', input_url)

    pusher = RecipePusher(recipe=recipe_text)
    
    fileNames = request.POST.getlist('files', [])
    pusher.fileTasksForInputs(input=fileNames)

    return HttpResponseRedirect('/status/queues/')


def view_svg(request):
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection
    from PYME.recipes import recipeLayout

    recipeURI = 'pyme-cluster:///' + request.GET.get('recipeURL').encode().lstrip('/')

    recipe = ModuleCollection.fromYAML(unifiedIO.read(recipeURI))

    svg = recipeLayout.to_svg(recipe.dependancyGraph())

    return HttpResponse(svg, content_type='image/svg+xml')

def extra_inputs(request):
    from PYME.IO import unifiedIO
    from PYME.recipes.modules import ModuleCollection

    recipeURI = 'pyme-cluster:///' + request.GET.get('recipeURL').encode().lstrip('/')

    recipe = ModuleCollection.fromYAML(unifiedIO.read(recipeURI))
    
    return render(request, 'recipes/extra_inputs.html', {'file_inputs': recipe.file_inputs})



