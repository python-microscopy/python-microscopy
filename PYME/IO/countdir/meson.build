
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/IO/countdir'

py_sources = files(
    '__init__.py',
    'setup.py',
)

py.install_sources(py_sources, subdir:'PYME/IO/countdir')

#FIXME - add_extension

if not (host_machine.system() == 'windows')
    py.extension_module(
        'countdir',
        [
            'countdir.c',
        ],
        include_directories: [np_include_dir],
        subdir: 'PYME/IO/countdir',
        dependencies: py.dependency(),
        c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
        install: true,
        #cpp_args: ['-std=c++17'],
    )
endif