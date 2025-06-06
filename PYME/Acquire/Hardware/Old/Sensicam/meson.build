
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Acquire/Hardware/Old/Sensicam'

py_sources = files(
    'sensicam.py',
    'sensicamMetadata.py',
    '__init__.py',
    'setup.py',
)

py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Old/Sensicam')

#FIXME - add_extension
