#!/usr/bin/python

##################
# setup.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/env python

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('cSMI', parent_package, top_path)

    config.add_extension('_cSMI',
        sources=["cSMI.i","DataStack.cpp","BaseRenderer.cpp","LUTRGBRenderer.cpp","DisplayParams.cpp","DisplayOpts.cpp","LineProfile.cpp"],
        include_dirs = [get_numpy_include_dirs(), '.'],
	extra_compile_args = ['-O3'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'Data display and kdf handling',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
Functions to handle data display, internal storage, and output in kdf format
''',
        license = "Proprietary",
	#options={'build_ext':{'swig_cpp':True, 'swig_opts':'-c++'}},
	#options={'swig_opts':'-c++'},
        **configuration(top_path='').todict())
