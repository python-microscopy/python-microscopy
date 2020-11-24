"""
Support for configuration / settings in PYME. This will eventually replace a motley collection of different ways of
storing the configuration in use in different parts of PYME. Configuration is read from the following locations (if they exist)
which are searched in order, with later entries over-riding prior ones:

 - an ``etc/PYME`` directory under the python ``sys.prefix``. This would usually be ``/usr/local/`` under standard unix
   python and the distribution base directory for a distribution such as anaconda. This allows installation wide
   configuration and is also a place that should be writable when installing conda packages and the like (useful for
   writing a package with plugins which register themselves - see plugin functions below.
 - a machine wide ``/etc/PYME`` directory. This is included for compatibility, but it is not envisaged that this will be used often, as it will only work on posix
   systems while the the above ``{{sys.prefix}}/etc/PYME`` option will also work under windows.
 - a ``.PYME`` directory under the users home directory. This should be used for user specific settings.

Within each configuration directory there can be a ``config.yaml`` file which stores configuration settings as key-value
pairs. These are accessed using the :func:`get` function.

The directories may also contain a ``plugins`` folder, which in turn can contain subfolders for ``visgui``, ``dsviewer``,
and ``recipes``.  PYME will also detect custom acquisition protocols saved in the ``.PYME/protocols`` directory,
similarly init scripts will be detected in ``.PYME/init_scripts`` directory. The
overall template for a configuration directory is as follows: ::

    .PYME
      |- config.yaml
      |- plugins
      |     |- visgui
      |     |     |- somemodule.txt
      |     |     |- anothermodule.txt
      |     |
      |     |- dsviewer
      |     |     |- somemodule.txt
      |     |
      |     |- recipes
      |     |      |- anothermodule.txt
      |     |
      |     |- <plugin-name>.yaml
      |     |- <another-plugin-name>.yaml
      |
      |- protocols
      |     |- a_protocol.py
      |     |- another_protocol.py
      |
      |- customrecipes
      |     |- myrecipe.yaml
      |     |- myOtherRecipe.yaml
      |
      |- init_scripts
            |- init_mymachine.py
            |- init_my_other_config.py
            
            
If you want to verify which directories will be searched on your particular python installation, running the following
on the command line will tell you:

.. code-block:: bash

    python -c "import PYME.config;print PYME.config.config_dirs"
    
with anaconda on OSX this gives me:

.. code-block:: bash

    ['/Users/david/.PYME', '/etc/PYME', '/Users/david/anaconda/etc/PYME']

Examples
========

config.yaml
-----------

The config.yaml file is essentially nothing more than a set of keys and values separated by colons. List and dictionary
parameter values are supported using standard yaml notation.

.. code-block:: yaml

    dataserver-root: "/Users/david/srvtest/test1"
    h5f-flush_interval: 1
    PYMEAcquire-extra_init_dir: "C:/pyme-init-scripts"



plugins/visgui/<plugin_name>.txt, plugins/dsview/<plugin_name>.txt, plugins/recipes/<plugin_name>.txt
-----------------------
If we were to use the plugin architecture to register some of the native visgui plugins (rather than using explicit
knowledge of their locations), the registration file would look something like this. Each line is a fully qualified
module import path.

::

    PYME.LMVis.Extras.photophysics
    PYME.LMVis.Extras.particleTracking

The same structure holds for dh5view plugins and dsview/<plugin_name>.txt and recipes/<plugin_name>.txt. NOTE - this
method of plugiin registration is supported for backwards compatibility only - new plugins should drop as single
<plugin-name>.yaml config file as detailed below.


plugins/<plugin-name>.yaml
--------------------------

A yaml file containing plugin information. This supersedes the previous separate plugin directories outlined above.
It should be formatted according to the following example. All sections are optional and may be omitted if the plugin
doesn't supply the features in question:

.. code-block:: yaml

    # a list of fully qualified module import paths for VisGUI plugins
    visgui:
        - somepackage.somemodule
        - somepackage.anothermodule
        
    # a list of fully qualified import paths for dh5view plugins
    dsviewer:
        - somepackage.somemodule
        
    # a list of fully qualified import paths for recipe modules
    recipes:
        - somepackage.somemodule
        
    # a section detailing templates and filters for jinga2 generated reports
    reports:
        # an importable module containing the templates
        # the module is just used to get the file path (i.e. we do `os.path.join(os.path.dirname(somemodule), template_name)`)
        # this module will get imported every time anything in PYME runs, so please make it lightweight (i.e. an
        # empty __init__.py in a dedicated templates folder which only has templates as the other contents)
        templates: somepackage.somemodule
        
        # jinga2 filters. Note that these will need to be pre-fixed by the plugin name when used in templates
        # to avoid name collisions with builtin filters or those from other plugins ie {{ value | myplugin.myfilter }}
        filters:
            somepackage.somemodule:
                - filter1
                - filter2
            somepagage.anothermodule:
                -filter3
                
    config:
        # any plugin specific config / settings which you want to put here - just a placeholder for now, but an implicit
        # promise that we won't clobber this key in the future.
        
    

In addition to the configuration derived from config.yaml, a few legacy environment variables are recognized. Subpackages
are also permitted to save configuration files in the ``.PYME`` directory.

Functions
=========


Known Config Keys
=================

This is a non-exhaustive list of configuration keys - if adding a new key, please add it here.

PYMEAcquire-extra_init_dir : default=None, a path to an extra directory to search for PYMEAcquire init files.

SplitterRatioDatabase : default=None, path to a .json formatted file containing information about ratiometric splitting ratios

VisGUI-console-startup-file : default=None, path to a script to run within the VisGUI interactive console on startup,
    used to populate the console namespace with additional functions. Note that this script should not manipulate the
    pipeline as this cannot be assumed to be loaded when the script runs.
    
dh5View-console-startup-file : default=None,   path to a script to run within the dh5View interactive console on startup,
    used to populate the console namespace with additional functions. Note that this script should not manipulate the
    data as this cannot be assumed to be loaded when the script runs.

TaskServer.process_queues_in_order : default = True, an optimisation for old style task distribution so that it processes
    series in the order that they were added to the queue (rather than the previous approach of choosing from a series
    at random). Means that caches behave sensibly, rather than potentially getting scrambled when multiple analyses run
    at once, but also means that you need to wait for previous series to finish before you get any results from the
    current series if you are doing real time analysis and not keeping up.

dataserver-root : what directory should PYMEDataSever serve. Overridden by the `--root` command line option. If undefined,
    the current directory is served. Note that this is also used in `clusterIO` to allow short-circuit access to data on
    the same node.
    
dataserver-filter : default = '', multi-cluster support. Specifies a cluster name / clusterfilter which identifies which
    cluster PYMEDataServer should identify with. First part of the PYME-CLUSTER url. Overridden by the `--server-filter`
    command line option.

dataserver-port : default=8080, what port to run the PYMEDataServer on. Overridden by the --port command line option (e.g. if you want to run multiple servers on one machine).

cluster-listing-no-countdir : default=False, hack to disable (True) the loading of the low-level countdir module which allows rapid
    directory statistics on posix systems. Needed on OSX if `dataserver-root` is a mapped network drive rather than a
    physical disk

clusterIO-hybridns : default=True, toggles whether a protocol compatibility 
    nameserver (True) or zeroconf only (False) is used in clusterIO. The hybrid
    nameserver offers greater protocol/version compatibility but is effectively
    two nameservers, which has performance implications on very high bandwidth 
    systems.

h5r-flush_interval : default=1, how often (in s) should we call the .flush() method and write records from the HDF/pytables
    caches to disk when writing h5r files.
    

nodeserver-chunksize : default=50, how many frames should we give a worker process at once (larger numbers = better
    background cache performance, but potentially not distributing as widely). Should be larger than the number of
    background frames when doing running average / percentile background subtraction [new style distribution].

nodeserver-worker-get-timeout : default=60, a timeout (in s). When the worker asks for tasks, the nodeserver tries to
    get nodeserver-chunksize tasks off its internal queue (which is filled by a separate thread which communicates with
    the ruleserver). This timeout specifies how long should the nodeserver wait when accessing this queue in the hope of
    finding a full chunk. If it times out, a partial chunk will be given to the worker. In practice, this timeout
    behaviour is responsible for clearing the small tail of tasks at the end of a series. [new-style distribution].

nodeserver-num_workers : default= CPU count. Number of workers to launch on an individual node.

ruleserver-retries : default = 3. [new-style task distribution]. The number of times to retry a given task before it is deemed to have failed.

httpspooler-chunksize : default=50, how many frames we spool in each chunk 
    before (potentially) switching which PYMEDataServer we send the next chunk
    to. Increasing the chunksize can increase data-locality for faster analysis,
    but has spooling/writing bandwidth implications.

pymevis-zoom-factor : default = 1.1, adjusts zoom sensitivity by adjusting magnification factor per scroll event


Deprecated config options
-------------------------

distributor-* : parameters for a previous implementation of cluster-based task distribution. Largely irrelevant now we use PYMERuleServer

VisGUI-new_layers : default = True, use the new-style layers view in VisGUI. Same as the --new_layers command line option.
    Largely a remnant of when I was running layers and old style VisGUI in parallel.

"""
import yaml
import os
import shutil
import sys
import glob

site_config_directory = '/etc/PYME'
site_config_file = '/etc/PYME/config.yaml'

dist_config_directory = os.path.join(sys.prefix, 'etc', 'PYME')
dist_config_file = os.path.join(dist_config_directory, 'config.yaml')

user_config_dir = os.path.join(os.path.expanduser('~'), '.PYME')
user_config_file = os.path.join(user_config_dir, 'config.yaml')

if not os.path.exists(user_config_dir):
    #if this is the first time we've called the module, make the config directory
    
    try:
        os.makedirs(user_config_dir)
    
        #touch our config file
        open(user_config_file, 'a').close()
    
        #copy template configuration files
        template_dir = os.path.join(os.path.split(__file__)[0], 'resources', 'config_template')
    
        conf_files = os.listdir(template_dir)
    
        for file in conf_files:
            shutil.copy(os.path.join(template_dir, file), os.path.join(user_config_dir, file))
    
        user_plugin_dir = os.path.join(user_config_dir, 'plugins')
        os.makedirs(user_plugin_dir)
        os.makedirs(os.path.join(user_plugin_dir, 'visgui'))
        os.makedirs(os.path.join(user_plugin_dir, 'dsviewer'))
        os.makedirs(os.path.join(user_plugin_dir, 'recipes'))
    except OSError:
        #we might not be able to write to the home directory
        pass

from PYME.IO.FileUtils import nameUtils

config_defaults = {
    'dataserver-root' : nameUtils.datadir,
}

config = {}
config.update(config_defaults)

config_dirs = [user_config_dir, site_config_directory, dist_config_directory]

for fn in [dist_config_file, site_config_file, user_config_file]:
    #loop over the three configuration locations and read files, if present.
    try:
        with open(fn) as f:
            dist_conf = yaml.safe_load(f)
            config.update(dist_conf)
    except (IOError, TypeError):
        #no configuration file found, or not formatted correctly
        pass


def get(key, default=None):
    """
    Gets a configuration parameter, by name

    Parameters
    ----------
    key : basestring
        The parameter name
    default : unspecified, optional
        The default value you want to assume if the parameter is undefined.

    Returns
    -------

    The parameter value, or the default value if undefined.

    """
    return config.get(key, default)

import logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# def get_plugins(application):
#     """
#     Get a list of plugins for a given application
#
#     Modules are registered by adding fully resolved module paths (one per line) to a text file in the relevant directory.
#     The code searches **all** files in the relevant directories, and the intention is that there is one registration file
#     for each standalone package that provides modules and can e.g. be conda or pip-installed which contains a list of all
#     the plugins that package provides. The registration filename should ideally be the same as the package name, although
#     further subdivision for large packages is fine. registration filenames should however be unique - e.g. by prefixing
#     with the package name. By structuring it this way, a package can add this file to the ``anaconda/etc/PYME/plugins/XXX/``
#     folder through the standard conda packaging tools and it will be automatically discovered without conflicts
#
#     Parameters
#     ----------
#     application : basestring
#         One of 'visgui', 'dsviewer', or 'recipes'
#
#     Returns
#     -------
#     list of fully resolved module paths
#
#     """
#     plugin_paths = []
#
#     for config_dir in config_dirs:
#         plugin_dir = os.path.join(config_dir, 'plugins', application)
#
#         try:
#             reg_files = glob.glob(os.path.join(plugin_dir, '*.txt'))
#         except OSError:
#             reg_files = []
#
#         for fn in reg_files:
#             with open(fn, 'r') as f:
#                 plugin_paths.extend(f.readlines())
#
#     logger.debug('plugin paths: ' +  str(plugin_paths))
#
#     return  list(set([p.strip() for p in plugin_paths if not p.strip() == '']))


report_template_dirs = {}
report_filters = {}
plugins = {}

def _parse_plugin_config():
    import importlib
    
    for config_dir in config_dirs:
        #parse the new style .yaml based config first
        plugin_yamls = glob.glob(os.path.join(config_dir, 'plugins','*.yaml'))
        
        for fn in plugin_yamls:
            plugin_name = os.path.splitext(os.path.basename(fn))[0]
            with open(fn, 'r') as f:
                plugin_conf = yaml.safe_load(f)
            
            for app in ['visgui', 'dsviewer', 'recipes']:
                plugins[app] = plugins.get(app, set()) | set(plugin_conf.get(app, []))
            
            try:
                # TODO - do we actually want to do the import here, or should we defer it to get_plugin_template_dirs()?
                # doing it here risks putting startup times for anything PYME related at the mercy of a badly written plugin
                report_template_dirs[plugin_name] = os.path.dirname(importlib.import_module(plugin_conf['reports']['templates']).__file__)
            except KeyError:
                pass
            except ImportError:
                logger.exception('Error finding templates for plugin %s' % plugin_name)
                
                
            try:
                report_filters[plugin_name] = plugin_conf['reports']['filters']
            except KeyError:
                pass
            
        #parse legacy .txt based plugin definitions
        def _get_app_txt_plugins(application):
            plugin_paths = []

            try:
                reg_files = glob.glob(os.path.join(config_dir, 'plugins', application, '*.txt'))
            except OSError:
                reg_files = []

            for fn in reg_files:
                with open(fn, 'r') as f:
                    plugin_paths.extend(f.readlines())

            return [p.strip() for p in plugin_paths if not p.strip() == '']
            

        for app in ['visgui', 'dsviewer', 'recipes']:
            plugins[app] = plugins.get(app, set()) | set(_get_app_txt_plugins(app))

_parse_plugin_config()
                
                
def get_plugin_report_filters():
    return report_filters.items()

def get_plugin_template_dirs():
    return report_template_dirs
            
def get_plugins(application):
    """
    Get a list of plugins for a given application

    Modules are registered by adding fully resolved module paths (one per line) to a text file in the relevant directory.
    The code searches **all** files in the relevant directories, and the intention is that there is one registration file
    for each standalone package that provides modules and can e.g. be conda or pip-installed which contains a list of all
    the plugins that package provides. The registration filename should ideally be the same as the package name, although
    further subdivision for large packages is fine. registration filenames should however be unique - e.g. by prefixing
    with the package name. By structuring it this way, a package can add this file to the ``anaconda/etc/PYME/plugins/XXX/``
    folder through the standard conda packaging tools and it will be automatically discovered without conflicts

    Parameters
    ----------
    application : basestring
        One of 'visgui', 'dsviewer', or 'recipes'

    Returns
    -------
    list of fully resolved module paths

    """
    return plugins[application]

def get_custom_protocols():
    """
    Get a dictionary recording the locations of any custom protocols.

    Returns
    -------

    A dictionary of {basename : full path} for any protocols found. In the current implementation
    custom protocols overwrite protocols of the same name in the PYME distribution.

    """
    import glob
    prots = {}
    for config_dir in config_dirs:
        prot_glob = os.path.join(config_dir, 'protocols/[a-zA-Z]*.py')
        prots.update({os.path.split(p)[-1] : p for p in glob.glob(prot_glob)})
    return prots

def get_custom_recipes():
    """
    Get a dictionary recording the locations of any custom recipes.

    Returns
    -------

    A dictionary of {basename : full path} for any recipes found.

    """
    import glob
    recipes = {}
    for config_dir in config_dirs:
        recip_glob = os.path.join(config_dir, 'customrecipes/[a-zA-Z]*.yaml')
        recipes.update({os.path.split(p)[-1] : p for p in glob.glob(recip_glob)})
    return recipes

def get_init_filename(filename, legacy_scripts_directory=None):
    """
    Look for an init file in the various locations. If the given filename exists (i.e. is a fully resolved path) it
    will be used. Otherwise 'init_scripts' subdirectory of the configuration directories will be searched, in order of
    precedence user - site - dist. It also checks for files in a provided directory (to support legacy usage with the
    PYMEAcquire/scripts directory) and the config option ``PYMEAcquire-extra_init_dir``.

    Parameters
    ----------

    filename: init file name to locate in script dirs

    Returns
    -------

    If found returns first match as full path to init file
    returns None if not found.

    """
    if os.path.exists(filename):
        return filename
    
    directories_to_search = [os.path.join(conf_dir, 'init_scripts') for conf_dir in config_dirs]
    
    extra_conf_dir = config.get('PYMEAcquire-extra_init_dir')
    if not extra_conf_dir is None:
        directories_to_search.insert(0, extra_conf_dir)
        
    if not legacy_scripts_directory is None:
        directories_to_search.insert(0, legacy_scripts_directory)
        

    for dir in directories_to_search:
        fnp = os.path.join(dir, filename)
        if os.path.exists(fnp):
            return fnp
        
    return None

def update_yaml_keys(fn, d, create_backup=False):
    """ 
    Update the keys in a YAML file without destroying comments.

    TODO: Plays it fast and lose with regex and won't work for many 
    situations. Currently only works in the case we have a line

    key : value  # optional comment, optional number of spaces
                 # around the colon, no option for a space at
                 # the beginning of the line
    
    To make this work well, we'll need a full YAML parser 
    that handles comments. We could either use ruamel.yaml 
    (https://pypi.org/project/ruamel.yaml/) or write our own.

    Parameters
    ----------
    fn : string
        Path to a yaml file.
    d : dict
        key, value pairs of keys/values to update or append to the
        end of the file
    create_backup : bool
        Make a backup of the YAML file before updating the keys
    """
    import re
    import json

    if create_backup:
        import shutil
        shutil.copy(fn, fn+'.bak')

    # Read the yaml file
    with open(fn) as f:
        data = f.read()

    # Update the appropriate keys
    for k, v in d.items():
        x = re.search(r'^{}\s*:.*$'.format(k),data,flags=re.MULTILINE)
        if x is None:
            data += '\n{}: {}'.format(k, json.dumps(v))
        else:
            data = re.sub(r'^{}\s*:.*$'.format(k),
                          '{}: {}'.format(k,json.dumps(v)),data,flags=re.MULTILINE)

    # Update the yaml file
    with open(fn, 'w') as f:
        f.write(data)

def update_config(d, config='user', config_fn='config.yaml', create_backup=False):
    """
    Updates PYME configuration files.

    Parameters
    ----------
    d : dict
        Dictionary of configuration keys to update
    config : str, optional
        PYME configuration type, one of ['user','site','dist'], by default 'user'
    config_fn : str, optional
        Name of the configuration file within the type, by default 'config.yaml'
    create_backup : bool
        Create a backup of the configuration file before updating.
    """
    
    # Choose where to look for the configuration file based
    # on the configuration type
    base = user_config_dir
    if config == 'site':
        base = site_config_directory
    elif config == 'dist':
        base = dist_config_directory
    
    # Open and edit the file
    update_yaml_keys(os.path.join(base,config_fn),d,create_backup=create_backup)

    # Reload config
    if sys.version_info.major == 3:
        from importlib import reload
    try:
        reload(sys.modules['config'])
    except(KeyError):
        reload(sys.modules['PYME.config'])
