
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/simulation/ChemDE'

py_sources = files(
    'fluorosceinQuench.py',
    'discreteReactions.py',
    'testReactions.py',
    'reactions.py',
    'fluorosceinQuench4.py',
    '__init__.py',
    'gillespie.py',
    'fluorosceinQuench2.py',
    'threeState.py',
    'setup.py',
    'fluorosceinQuench3.py',
    'thiolBlink.py',
)

py.install_sources(py_sources, subdir:'PYME/simulation/ChemDE')


#FIXME - add_extension
#FIXME - add_extension

py.extension_module(
    'StateMC',
    [
        'StateMonteCarlo.c',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/simulation/ChemDE',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)

py.extension_module(
    'contEvents',
    [
        'contEvents.c',
    ],
    
    include_directories: [np_include_dir],
    subdir: 'PYME/simulation/ChemDE',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)