import os

qhullSources = ['user.c', 'global.c', 'stat.c', 'io.c', 'geom2.c', 'poly2.c',
       'merge.c', 'geom.c', 'poly.c', 'qset.c', 'mem.c', 'usermem.c', 'userprintf.c', 'rboxlib.c','random.c','libqhull.c']

qhullSources = ['qhull/' + s for s in qhullSources]

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs, yellow_text
    config = Configuration('qHull', parent_package, top_path)

    #print 'foo'
    #print yellow_text('foo' + config.local_path)

    srcs = ['triangWrap.c', 'triangRend.c', '../SoftRend/drawTriang.c']

    #check for drift correction code
    if os.path.exists(os.path.join(config.local_path, '../DriftCorrection/triangLhood.c')):
        print yellow_text('compiling with drift correction')
        srcs.append('../DriftCorrection/triangLhood.c')
    else:
        print yellow_text('compiling without drift correction')
        srcs.append('lhoodStubs.c')

    config.add_extension('triangWrap',
        sources=srcs + qhullSources,
        include_dirs = get_numpy_include_dirs()+['qhull', '../SoftRend'],
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
