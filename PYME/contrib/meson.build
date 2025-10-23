
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/contrib'

py_sources = files(
    'congrid.py',
    'TextCtrlAutoComplete.py',
    '__init__.py',
    'listctrlMixins.py',
    'auiFloatBook.py',
    'setup.py',
    'lzw.py',
    'wxPlotPanel.py',
)

py.install_sources(py_sources, subdir:'PYME/contrib')

py_sources = files(
    'pad/__init__.py',
    'pad/info_pad.py',
    'pad/pad.py',
)

py.install_sources(py_sources, subdir:'PYME/contrib/pad')

subdir('gohlke')

py_sources = files(
    'dispatch/dispatcher.py',
    'dispatch/__init__.py',
)

py.install_sources(py_sources, subdir:'PYME/contrib/dispatch')

py_sources = files(
    'aniso/aniso.py',
    'aniso/__init__.py',
)

py.install_sources(py_sources, subdir:'PYME/contrib/aniso')

