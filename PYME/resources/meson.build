
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/resources'

py_sources = files(
    '__init__.py',
    'setup.py',
)

py.install_sources(py_sources, subdir:'PYME/resources')

install_subdir('icons', install_dir: 'PYME/resources/icons')
install_subdir('config_template', install_dir: 'PYME/resources/config_template')
install_subdir('web', install_dir: 'PYME/resources/web')
install_subdir('test_datasets', install_dir: 'PYME/resources/test_datasets')
