#!/usr/bin/env python

def configuration(parent_package='', parent_path=None):

    # The following three lines constitute minimal contents
    # of configuration(..) that is suitable for pure Python
    # packages.
    package = 'pad'
    from scipy_distutils.misc_util import default_config_dict
    config = default_config_dict(package,parent_package)

    return config

if __name__ == '__main__':
    from scipy_distutils.core import setup
    setup(**configuration())
