def create_pymeimage_plugin():
    plugin_template = f'''\
# This is a template for a PYMEImage UI plugin.
from PYME.DSView.modules._base import Plugin

class MyCoolPluginKlass(Plugin):
    def __init__(self, dsviewer):
        super().__init__(dsviewer)

        # Add your code here

        # Example: Add a menu item
        dsviewer.AddMenuItem('Processing', 'Do something neat', self.OnDoSomethingNeat)

    def OnDoSomethingNeat(self, event=None):
        # Menu item callback
        # Add your code here
        pass
    
def Plug(dsviewer):
    """ This function is called by PYMEImage to create an instance of the plugin """
    return MyCoolPluginKlass(dsviewer)
'''
    return plugin_template
    

def create_pymevis_plugin():
    plugin_template = f'''\
# This is a template for a PYMEVisualize UI plugin.

def DoSomethingNeat(vis_frame):
    """ Example function that does something neat, takes a reference
    to the main PYMEvis window as an argument, which permits access to
    the pipeline and recipe objects, as well as the main window itself and
    the open GL canvas"""
    
    pipeline = vis_frame.pipeline # the PYME.LMVis.pipeline.Pipeline object associated with the window
    recipe = pipeline.recipe # the current recipe
    canvas = vis_frame.gl_canvas # the open GL canvas - an instance of PYME.LMVis.gl_render3D_shaders.LMGLShaderCanvas

    # Add your code here
    pass
    
def Plug(vis_frame):
    """ This function is called by PYMEVisualize to create an instance of the plugin """
    # Add your code here
    
    pass
    
    # Example: Add a menu item
    vis_frame.AddMenuItem('My neat menu name', 'Do something neat', lambda e : DoSomethingNeat(vis_frame))
    
'''
    return plugin_template

def create_recipe_plugin():
    plugin_template = f'''\
# this is a template to create a recipe plugin with one or more modules
from PYME.recipes.base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, FileOrURI, CInt

import logging
logger=logging.getLogger(__name__)

@register_module('MyCoolGenericModule')
class MyCoolGenericModule(ModuleBase):
    """ This is a template for a recipe module """
    
    # Define the inputs and outputs for the module
    input = Input('input')
    optional_input = Input('optional_input'))
    output = Output('output')
    
    # Define the parameters for the module
    param1 = Float(1.0, description='A parameter')
    param2 = Int(1, description='Another parameter')
    
    def run(self, input, optional_input=None):
        """Implement a .run() method to perform the module's operation.
        
        - The argument names must match those of the defined Inputs
        - If there is more than one Output, return a dictionary with the output names as keys
        - Recipe modules should be side-effect free (i.e. they should not modify 
          the input data, nor remember computed "state" between runs)
        - Recipe modules are an interface to code, and should be fairly lightweight 
          (i.e. if the run() method ends up being more than about 20 lines, consider 
          moving the algorithmic logic into a separate file and calling that from run() 
          - this ends up much more readable)
        """

        # Add your code here

        result = something_cool(input, optional_input, self.param1, self.param2) 
        
        return result



@register_module('MyCoolImageFilter')
class MyCoolImageFilter(Filter):
    """ This is a template for a recipe module in which both input and output are images 
    It uses the Filter class to avoid having to mess with all the boiler plate involved in converting
    data in ImageStacks to and fron numpy arrays.

    Instead of implementing run(), derived classes should implement an apply_filter() method that takes
    a numpy array as input and returns a numpy array as output.

    The Input and Output traits are defined in the base class, along with a 'dimensionality' trait
    which allows the user to tell the module what dimensions of the data it should operate on.
    If dimensionalality is `XY`, apply_filter will get a 2D array, and be called for each slice, 
    timepoint, and channel in the input data. If it is `XYZ`, apply_filter will get a 3D array, and 
    be called for each timepoint and channel in the input data. If it is `XYZT`, apply_filter will get 
    a 4D array, and be called for each channel in the input data. If a module cannot handle higher
    dimensional data, it should re-define the dimensionality trait to only offer the options it can handle.
    """

    # Define the parameters for the module
    param1 = Float(1.0, description='A parameter')

    #redifine the dimensionality trait to only handle 2D and 3D data
    dimensionality = Enum(['XY', 'XYZ'], default_value='XY')

    def apply_filter(self, data, voxelsize):
        """Implement an apply_filter() method to perform the module's operation.
        
        - data is a numpy array with the data to be processed
        - voxelsize is a named tuple with the voxel sizes in each dimension (x, y, and z) in nm
        - The above caveats about being side-effect free and lightweight apply here too.
        """

        # Add your code here

        result = something_cool(data, self.param1) 
        
        return result
'''
    return plugin_template

def create_fit_factory_plugin():
    plugin_template = f'''\
# This is a template for a fit factory plugin which re-implements the GaussianFitFactory, albeit with a
# slow, completely python implementation of the fit function.

import numpy as np
from .fitCommon import fmtSlicesUsed 
from . import FFBase
from PYME.Analysis._fithelpers import FitModelWeighted 


##################
# Model function
def f_gauss2dSlow(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    return A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*(X -x0) + b_y*(Y-y0)


#####################
#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('sigma', '<f4'), 
                              ('background', '<f4'),
                              ('bx', '<f4'),
                              ('by', '<f4')]),
              ('fitError', [('A', '<f4'),
                            ('x0', '<f4'),
                            ('y0', '<f4'),
                            ('sigma', '<f4'), 
                            ('background', '<f4'),
                            ('bx', '<f4'),
                            ('by', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4'),
              ('nchi2', '<f4')
              ]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0, nchi2=-1):
    slicesUsed = fmtSlicesUsed(slicesUsed)  
    res = np.zeros(1, dtype=fresultdtype)
    
    n_params = len(fitResults)
    res['tIndex'] = metadata['tIndex']
    res['fitResults'].view('7f4')[:n_params] = fitResults

    if fitErr is None:
        res['fitError'].view('7f4')[:] = -5e3
    else:
        res['fitError'].view('7f4')[:n_params] = fitErr
        
    res['resultCode'] = resultCode
    res['slicesUsed'] = slicesUsed
    res['subtractedBackground'] = background
        
    res['nchi2'] = nchi2
    
    return res


class SlowGaussianFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_gauss2dSlow, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma, **kwargs)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background

        #print dataMean.min(), dataMean.max()

        #estimate some start parameters...
        A = data.max() - data.min() #amplitude

        vs = self.metadata.voxelsize_nm
        x0 =  vs.x*x
        y0 =  vs.y*y
        
        bgm = np.mean(background)
        startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]

        #do the fit
        (res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
                

        #try to estimate errors based on the covariance matrix
        fitErrors=None
        try:       
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception:
            pass

        nchi2 = (infodict['fvec']**2).sum()/(data.size - res.size)
        #package results
        return GaussianFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm, nchi2)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*np.mgrid[(y - roiHalfSize):(y + roiHalfSize + 1)]

        return (f_gauss2dSlow(params, X, Y), X[0], Y[0], 0)


#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.IntParam('Analysis.ROISize', u'ROI half size', 5),

]

DESCRIPTION = 'Slow version of vanilla 2D Gaussian fit.'
LONG_DESCRIPTION = 'Slow version of single colour 2D Gaussian fit, used as a template for creating new fit factories.'
USE_FOR = '2D single-colour'


'''
    return plugin_template

#def create_reports_readme():

def create_install_script(package_name):
    install_script = f'''\
from PYME import config
import os
import sys
import shutil

def install_plugin(dist=False):
    this_dir = os.path.dirname(__file__)

    if dist:
        shutil.copyfile(os.path.join(this_dir, '{package_name}.yaml'), 
                        os.path.join(config.dist_config_directory, 'plugins', {package_name}.yaml'))
    else:  #default to user config directory
        shutil.copyfile(os.path.join(this_dir, '{package_name}.yaml'), 
                        os.path.join(config.user_config_dir, 'plugins', {package_name}.yaml'))

if __name__ == '__main__':
    import sys
    if (len(sys.argv) >= 1) and (sys.argv[1] == 'dist'):
        dist = True
    else:
        dist = False

    install_plugin(dist)
'''
    return install_script

def create_setup_script(package_name):
    setup_script = f'''\
#!/usr/bin/env python

# Fill out this template to enable setuptools installation of your plugin as a
# Python package, i.e. `python setup.py develop` or `python setup.py install`.
# This script it set up to additionally register the plugin with PYME in a post-
# install command by running the included `install_plugin.py` script. You can
# get arbitrarily fancy there if you like in terms of configuring your plugin.

# Replace 'package_name' with the name of your plugin. This must be importable,
# and will point towards the directory by the same name in your repository. For
# example you might call your repository directory 'pyme-plugin' but the
# directory inside of it which stores all of your installable python code should
# be called 'pyme_plugin'.
PACKAGE_NAME = '{package_name}'
# Set your version, as a string. Something like YY.MM.DD works well here
PACKAGE_VERSION = '00.00.00'
# Include a short description of your package. This might eventually get
# displayed in e.g. Anaconda cloud if you build/upload packages, etc.
PACKAGE_DESCRIPTION = 'What your plugin does'

# -------- If you filled in everything up to here you should be set ------------

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_pyme_plugin():
    import sys
    import subprocess
    import os
    plugin_install_path = os.path.join(os.path.dirname(__file__),
                                       'install_plugin.py')
    subprocess.Popen('%s %s' % (sys.executable, plugin_install_path), 
                        shell=True)


class DevelopModuleAndInstallPlugin(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_pyme_plugin()
        

class InstallModuleAndInstallPlugin(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_pyme_plugin()


setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description=PACKAGE_DESCRIPTION,
    packages=find_packages(),
    cmdclass={{
        'develop': DevelopModuleAndInstallPlugin,
        'install': InstallModuleAndInstallPlugin,
    }},
)
'''
    return setup_script   

def create_pyme_plugin_template(output_dir, package_name, pymeimage=True, pymevis=True, recipe=True, fit_factories=False, reports=False):
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)

    mod_dir = os.path.join(output_dir, package_name)
    os.makedirs(mod_dir, exist_ok=True)
    Path(os.path.join(mod_dir, '__init__.py')).touch()

    plugin_conf = {}

    recipe_mod_dir = os.path.join(mod_dir, 'recipe_modules')

    if pymeimage:
        ui_mod_dir = os.path.join(mod_dir, 'dsview_plugins')
        os.makedirs(ui_mod_dir, exist_ok=True)
        Path(os.path.join(ui_mod_dir, '__init__.py')).touch()

        plugin_conf['dsviewer'] = []

        if not isinstance(pymeimage, str):
            pymeimage = package_name

        for p in pymeimage.split(','):
            with open(os.path.join(ui_mod_dir, f'{p}.py'), 'w') as f:
                f.write(create_pymeimage_plugin())

            plugin_conf['dsviewer'].append(f'{package_name}.dsview_plugins.{p}')

    if pymevis:
        #  we should be able to put all the PYMEvis plugins for any package in a single file.
        with open(os.path.join(mod_dir, 'pymevis_plugins.py'), 'w') as f:
            f.write(create_pymevis_plugin())

        plugin_conf['visgui'] = [f'{package_name}.pymevis_plugins',]

    if recipe:
        os.makedirs(recipe_mod_dir, exist_ok=True)
        Path(os.path.join(recipe_mod_dir, '__init__.py')).touch()

        with open(os.path.join(recipe_mod_dir, f'{package_name}.py'), 'w') as f:
            f.write(create_recipe_plugin())

        plugin_conf['recipe_modules'] = [f'{package_name}.recipe_modules.{package_name}',]

    if fit_factories:
        os.makedirs(os.path.join(mod_dir, 'fit_factories'), exist_ok=True)
        Path(os.path.join(mod_dir, 'fit_factories', '__init__.py')).touch()
        with open(os.path.join(mod_dir, 'fit_factories', f'myfitfactory.py'), 'w') as f:
            f.write(create_fit_factory_plugin())

        plugin_conf['fit_factories'] = [f'{package_name}.fit_factories.myfitfactory',]

    if reports:
        os.makedirs(os.path.join(mod_dir, 'reports'), exist_ok=True)
        Path(os.path.join(mod_dir, 'reports', '__init__.py')).touch()
        Path(os.path.join(mod_dir, 'reports', 'filters.py')).touch()
        os.makedirs(os.path.join(mod_dir, 'reports', 'templates'), exist_ok=True)
        Path(os.path.join(mod_dir, 'reports', 'templates', '__init__.py')).touch()

        plugin_conf['reports'] = {'templates': f'{package_name}.reports.templates', 
                                  'filters': [{f'{package_name}.reports.filters':'my_filter'},]}
        

    import yaml
    with open(os.path.join(output_dir, f'{package_name}.yaml'), 'w') as f:
        yaml.safe_dump(plugin_conf, f)

    with open(os.path.join(output_dir, 'install_plugin.py'), 'w') as f:
        f.write(create_install_script(package_name))

    with open(os.path.join(output_dir, 'setup.py'), 'w') as f:
        f.write(create_setup_script(package_name))

    

        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a template for a PYME plugin')
    parser.add_argument('output_dir', help='Directory to write the plugin template to')
    parser.add_argument('package_name', help='Name of the package')
    parser.add_argument('--pymeimage', help='Create a PYMEImage plugin with the given name', default=True)
    parser.add_argument('--pymevis', help='Create a PYMEVisualize plugin with the given name', default=True)
    parser.add_argument('--recipe', help='Create a recipe module with the given name', default=True)
    parser.add_argument('--fit_factories', help='Create a fit factory with the given name', default=True)
    parser.add_argument('--reports', help='Create a reports module with the given name', default=False)
    args = parser.parse_args()

    create_pyme_plugin_template(args.output_dir, args.package_name, args.pymeimage, args.pymevis, args.recipe, args.fit_factories, args.reports)

    
    