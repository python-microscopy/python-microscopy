import sys

if sys.platform == 'darwin':#MacOS
    linkArgs = []
else:
    linkArgs = ['-static-libgcc']
	

#windows VC++ has really shocking c standard support so we need to include
#custom stdint.h and intypes.h files from https://code.google.com/archive/p/msinttypes
if sys.platform == 'win32':
	extra_include_dirs = ['win_incl']
else:
	extra_include_dirs = []

#from PYME.misc import cython_numpy_monkey

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('bcl', parent_package, top_path)

    config.add_extension('bcl',
        sources=['bcl.pyx', 'src/huffman.c'],
        include_dirs = ['src', get_numpy_include_dirs()] + extra_include_dirs,
	extra_compile_args = ['-O3', '-fno-exceptions', '-ffast-math', '-march=native', '-mtune=native'],
        extra_link_args=linkArgs)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'python wrapper for BCL',
    	author = 'David Baddeley',
       	author_email = 'david.baddeley@yale.edu',
       	url = '',
       	long_description = """
Python wrapper for the Basic compression libarary
""",
          license = "BSD",
          **configuration(top_path='').todict()
          )

