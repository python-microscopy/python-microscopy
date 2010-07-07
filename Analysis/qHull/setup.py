qhullSources = ['user.c', 'global.c', 'stat.c', 'io.c', 'geom2.c', 'poly2.c',
       'merge.c', 'geom.c', 'poly.c', 'qset.c', 'mem.c', 'usermem.c', 'userprintf.c', 'rboxlib.c','random.c','libqhull.c']

qhullSources = ['qhull/' + s for s in qhullSources]

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('triangWrap', parent_package, top_path)

    config.add_extension('triangWrap',
        sources=['triangWrap.c', 'triangLhood.c'] + qhullSources,
        include_dirs = get_numpy_include_dirs()+['qhull'],
	extra_compile_args = ['-O3', '-fno-exceptions', '-ffast-math'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'qhull wrapper',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
qhull wrapper for various triangularsiation functions
''',
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
