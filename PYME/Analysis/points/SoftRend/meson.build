
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
##py = import('python').find_installation(pure: false)
##np_include_dir = run_command(py,  ['-c',  'import numpy; print(numpy.get_include())'], check: true).stdout().strip()



install_dir = py.get_install_dir() / 'PYME/Analysis/points/SoftRend'

py_sources = files(
    '__init__.py',
    'setup.py',
    'testTetrahedra.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/points/SoftRend')

#FIXME - add_extension

qhullSources = ['qhull/user.c', 'qhull/global.c', 'qhull/stat.c', 'qhull/io.c', 'qhull/geom2.c', 'qhull/poly2.c',
       'qhull/merge.c', 'qhull/geom.c', 'qhull/poly.c', 'qhull/qset.c', 'qhull/mem.c', 'qhull/usermem.c', 'qhull/userprintf.c', 'qhull/rboxlib.c','qhull/random.c','qhull/libqhull.c']

#qhullSources = ['qhull/' + s for s in qhullSources]

srcs = ['triRend.c','drawTriang.c', 'triangRend.c']

#message('numpy include dir:', np_include_dir)

py.extension_module(
    'triRend',
    srcs + qhullSources,
    include_directories: [np_include_dir, 'qhull'],
    subdir: 'PYME/Analysis/points/SoftRend',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math', '-Wno-unused-but-set-variable'],
    install: true,
    #cpp_args: ['-std=c++17'],
)
   