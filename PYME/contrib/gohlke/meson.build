
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/contrib/gohlke'

py_sources = files(
    'tifffile.py',
    'czifile.py',
    '__init__.py',
    'setup.py',
    'fcsfiles.py',
)

py.install_sources(py_sources, subdir:'PYME/contrib/gohlke')


#FIXME - add_extension
py.extension_module(
    '_tifffile',
    [
        'tifffile.c',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/contrib/gohlke',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)