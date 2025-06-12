
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/recipes'

py_sources = files(
    'simulation.py',
    'vertical_recipe_display.py',
    'recipeLayout.py',
    'pyramid.py',
    'runRecipe.py',
    'multiview.py',
    'batchProcess.py',
    'bakeshop.py',
    'measurement.py',
    '__init__.py',
    'tablefilters.py',
    'fudges.py',
    'localisations.py',
    'graphing.py',
    'recipeGui.py',
    'html_recipe_editor.py',
    'setup.py',
    'surface_fitting.py',
    'pointcloud.py',
    'skfilters.py',
    'machine_learning.py',
    'traits.py',
    'modules.py',
    'acquisition.py',
    'processing.py',
    'recipe.py',
    'tracking.py',
    'base.py',
    'filters.py',
    'output.py',
)

py.install_sources(py_sources, subdir:'PYME/recipes')

install_subdir('Recipes', install_dir: install_dir)
install_subdir('templates', install_dir: install_dir)
