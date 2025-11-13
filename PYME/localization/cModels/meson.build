
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/localization/cModels'

py_sources = files(
    '__init__.py',
)

py.install_sources(py_sources, subdir:'PYME/localization/cModels')

#FIXME - add_extension
py.extension_module(
    'gauss_app',
    [
        'gauss_ap.c',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/localization/cModels',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)