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

user_config_dir = os.path.join(os.path.expanduser('~'), '.PYME')
user_config_file = os.path.join(user_config_dir, 'config.yaml')

config_defaults = {}

config = {}
config.update(config_defaults)

#read the per-user configuration
try:
    with open(user_config_file) as f:
        user_conf = yaml.load(f)
        config.update(user_conf)

except IOError:
    #no local configuration file found
    pass