
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Acquire/Hardware/Simulator'

py_sources = files(
    'fakePiezo.py',
    'simcontrol.py',
    '__init__.py',
    'fakeCam.py',
    'lasersliders.py',
    'dSimControl.py',
    'fluor.py',
    'setup.py',
    'simui_wx.py',
    'EmpiricalHist.py',
    'rend_im.py',
)

py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Simulator')

data_files = files(
    'illuminate.pyx',
)

install_data(data_files, install_dir: install_dir)

#FIXME - Extension()
py.extension_module(
    'illuminate',
    [
        'illuminate.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/Acquire/Hardware/Simulator',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)