
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Analysis/points/traveling_salesperson'

py_sources = files(
    'sectioned_two_opt.py',
    'sort.py',
    'two_opt.py',
    '__init__.py',
    'setup.py',
    'queue_opt.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/points/traveling_salesperson')

data_files = files(
    'two_opt_utils.pyx',
)

install_data(data_files, install_dir: install_dir)

#FIXME - Extension()

py.extension_module(
    'two_opt_utils',
    [
        'two_opt_utils.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/Analysis/points/traveling_salesperson',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)