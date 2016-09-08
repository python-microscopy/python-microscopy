"""
Support for configuration / settings in PYME. This will eventually replace a motley collection of different ways of
storing the configuration in use in different parts of PYME. Configuration is a simple key-value dictionay and is
stored in .pymerc.yaml files. The following locations are searched in order:

 - the directory containing this file [defaults / site-wide]
 - the users home directory

 In addition, a few legacy environment variables are recognized.
"""
import yaml
import os
import shutil

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





config_defaults = {}

config = {}
config.update(config_defaults)

#read the per-user configuration
try:
    with open(user_config_file) as f:
        user_conf = yaml.load(f)
        config.update(user_conf)

except (IOError, TypeError):
    #no local configuration file found, or not formatted correctly
    pass


def get(key, default=None):
    return config.get(key, default)