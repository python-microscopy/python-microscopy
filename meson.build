project(
  'python-microscopy',
  'c', 'cython',
  version: run_command(
     # This should become `numpy/_version.py` in NumPy 2.0
     ['python',  './PYME/version.py'],
     check: true).stdout().strip(),
  license: 'GNU General Public License v3.0',
  meson_version: '>=1.2.99',  # version in vendored-meson is 1.2.99
  
#   default_options: [
#     'buildtype=debugoptimized',
#     'b_ndebug=if-release',
#     'c_std=c11',
#     'cpp_std=c++17',
#     'pkgconfig.relocatable=true',
#   ],
)

py = import('python').find_installation(pure: false)
np_include_dir = include_directories(run_command(py,  ['-c',  'import numpy; print(numpy.get_include())'], check: true).stdout().strip())
#message('numpy include dir:', np_include_dir)
message('install_dir:', py.get_install_dir())

add_global_arguments(
 #'-fcompare-debug-secondary',
 #'-wno'
 '-ffast-math',
 '-Wno-unused-parameter',
 '-Wno-unused-variable',
 '-Wno-unused-function',
 language: 'c'
 )



subdir('PYME')