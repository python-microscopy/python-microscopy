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
and ``recipes``.  PYME will also detect custom acquisition protocols saved in the ``.PYME/protocols`` directory. the
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
      |           |- anothermodule.txt
      |
      |- protocols
            |- a_protocol.py
            |- another_protocol.py


Examples
========

config.yaml
-----------

The config.yaml file is essentially nothing more than a set of keys and values separated by colons. List and dictionary
parameter values are supported using standard yaml notation.

.. code-block:: yaml

    dataserver-root: "/Users/david/srvtest/test1"
    h5f-flush_interval: 1


plugins/visgui/PYME.txt
-----------------------
If we were to use the plugin architecture to register some of the native plugins (rather than using explicit knowledge
of their locations), the registration file would look something like this. Each line is a fully qualified module import
path.

::

    PYME.LMVis.Extras.photophysics
    PYME.LMVis.Extras.particleTracking

In addition to the configuration derived from config.yaml, a few legacy environment variables are recognized. Subpackages
are also permitted to save configuration files in the ``.PYME`` directory.

Functions
=========
"""
import yaml
import os
import shutil
import sys

site_config_directory = '/etc/PYME'
site_config_file = '/etc/PYME/config.yaml'

dist_config_directory = os.path.join(sys.prefix, 'etc', 'PYME')
dist_config_file = os.path.join(dist_config_directory, 'config.yaml')

user_config_dir = os.path.join(os.path.expanduser('~'), '.PYME')
user_config_file = os.path.join(user_config_dir, 'config.yaml')

if not os.path.exists(user_config_dir):
    #if this is the first time we've called the module, make the config directory
    os.makedirs(user_config_dir)

    #touch our config file
    open(user_config_file, 'a').close()

    #copy template configuration files
    template_dir = os.path.join(os.path.split(__file__)[0], 'resources', 'config_template')

    conf_files = os.listdir(template_dir)

    for file in conf_files:
        shutil.copy(os.path.join(template_dir, file), os.path.join(user_config_dir, file))

    user_plugin_dir = os.path.join(user_config_dir, 'plugins')
    # os.makedirs(user_plugin_dir, exist_ok=True) # exist_ok is python 3.X
    try:
        os.makedirs(user_plugin_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    os.makedirs(os.path.join(user_plugin_dir, 'visgui'))
    os.makedirs(os.path.join(user_plugin_dir, 'dsviewer'))
    os.makedirs(os.path.join(user_plugin_dir, 'recipes'))


config_defaults = {}

config = {}
config.update(config_defaults)

for fn in [dist_config_file, site_config_file, user_config_file]:
    #loop over the three configuration locations and read files, if present.
    try:
        with open(fn) as f:
            dist_conf = yaml.load(f)
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
    plugin_paths = []

    for config_dir in [dist_config_directory, site_config_directory, user_config_dir]:
        plugin_dir = os.path.join(config_dir, 'plugins', application)

        try:
            reg_files = os.listdir(plugin_dir)
        except OSError:
            reg_files = []

        for fn in reg_files:
            with open(fn, 'r') as f:
                plugin_paths.append(f.readlines())

    return  list(set([p.strip() for p in plugin_paths if not p.strip() == '']))


def get_custom_protocols():
    """
    Get a dictionary recording the locations of any custom protocols.

    Returns
    -------

    A dictionary of {basename : full path} for any protocols found.

    """
    import glob
    prots = {}
    for config_dir in [dist_config_directory, site_config_directory, user_config_dir]:
        prot_glob = os.path.join(config_dir, 'protocols/[a-zA-Z]*.py')
        prots.update({os.path.split(p)[-1] : p for p in glob.glob(prot_glob)})





