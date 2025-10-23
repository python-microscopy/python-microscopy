
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
py = import('python').find_installation(pure: false)
np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/contrib/cpmath'

py_sources = files(
    'radial_power_spectrum.py',
    'convex_hull.py',
    'princomp.py',
    'zernike.py',
    'index.py',
    'rankorder.py',
    'bg_compensate.py',
    '__init__.py',
    'cpmorphology.py',
    'threshold.py',
    'watershed.py',
    'setup.py',
    'outline.py',
    'smooth.py',
    'filter.py',
    'haralick.py',
    'otsu.py',
    'propagate.py',
    'lapjv.py',
)

py.install_sources(py_sources, subdir:'PYME/contrib/cpmath')

data_files = files(
    '_propagate.pyx',
    '_lapjv.pyx',
    '_cpmorphology2.pyx',
    '_filter.pyx',
    '_convex_hull.pyx',
    '_watershed.pyx',
)

install_data(data_files, install_dir: 'PYME/contrib/cpmath')
FIXME - Extension()FIXME - Extension()FIXME - Extension()FIXME - Extension()FIXME - Extension()FIXME - Extension()FIXME - Extension()