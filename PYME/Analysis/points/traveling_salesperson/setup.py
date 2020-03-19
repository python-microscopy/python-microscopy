

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('astigmatism', parent_package, top_path)

    config.add_extension('two_opt_utils',
        sources=['two_opt_utils.c'],
        include_dirs = [get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(
          **configuration(top_path='').todict()
          )
