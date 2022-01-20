
import sys
import os

if sys.platform == 'darwin'  :  # MacOS
    link_args = []
else:
    link_args = ['-static-libgcc']

# windows VC++ has really shocking c standard support so we need to include
# custom stdint.h and intypes.h files from https://code.google.com/archive/p/msinttypes
# print os.environ.get('CC', 'foo')
if sys.platform == 'win32' and not os.environ.get('CC', '') == 'mingw':
    extra_include_dirs = ['win_incl']
else:
    extra_include_dirs = []

from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):

    from numpy.distutils.core import Extension
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    cur_dir = os.path.dirname(__file__)

    ext = Extension(name='.'.join([parent_package, 'traveling_salesperson', 'two_opt_utils']),
                    sources=[os.path.join(cur_dir, 'two_opt_utils.pyx')],
                    include_dirs= get_numpy_include_dirs() + extra_include_dirs,
                    extra_compile_args=['-O3', '-fno-exceptions', '-ffast-math'],#, '-march=native', '-mtune=native'],
                    extra_link_args=link_args)

    config = Configuration('traveling_salesperson', parent_package, top_path, ext_modules=cythonize([ext]))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(
          **configuration(top_path='').todict()
          )
