
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/resources'

py_sources = files(
    '__init__.py',
)

py.install_sources(py_sources, subdir:'PYME/resources')

#install_subdir('icons', install_dir: install_dir / 'icons', strip_directory: true)
icon_files = run_command(py, ['-c', 'import glob; print(" ".join(glob.glob("icons/*.*")))'], check: true).stdout().strip().split()
py.install_sources(icon_files, subdir: 'PYME/resources/icons')
install_subdir('config_template', install_dir: install_dir  / 'config_template', strip_directory: true)

install_subdir('web', install_dir: install_dir )
#install_subdir('test_datasets', install_dir: install_dir )
dataset_files = run_command(py, ['-c', 'import glob; print(" ".join(glob.glob("test_datasets/*.*")))'], check: true).stdout().strip().split()
py.install_sources(dataset_files, subdir: 'PYME/resources/test_datasets')
