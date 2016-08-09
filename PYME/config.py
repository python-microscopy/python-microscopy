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

config_defaults = {}