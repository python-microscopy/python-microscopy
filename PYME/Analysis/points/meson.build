
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Analysis/points'

py_sources = files(
    'spherical_harmonics.py',
    'gen3DTriangs.py',
    'ripleys.py',
    'twoColour.py',
    'multiview.py',
    'moments.py',
    '__init__.py',
    'tCluster.py',
    'qpaint.py',
    'cluster_morphology.py',
    'coordinate_tools.py',
    'setup.py',
    'fiducials.py',
    'objectMeasure.py',
    'segmentLL.py',
    'rings.py',
    'surfit.py',
    'twoColourPlot.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/points')

py_sources = files(
    'QuadTree/qtUtils.py',
    'QuadTree/pointQT.py',
    'QuadTree/QTrend.py',
    'QuadTree/__init__.py',
    'QuadTree/pointQTF.py',
    'QuadTree/rendQT.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/points/QuadTree')

subdir('SoftRend')
subdir('DistHist')
subdir('DeClump')
subdir('EdgeDB')
subdir('arcfit')
subdir('astigmatism')
subdir('traveling_salesperson')
subdir('features')
