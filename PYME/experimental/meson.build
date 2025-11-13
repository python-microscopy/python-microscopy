
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/experimental'

py_sources = files(
    'aqavg.py',
    'tcHist.py',
    'expectationMaximisation.py',
    'treap.py',
    'fusionUtils.py',
    'fusionRadial.py',
    'dual_marching_cubes.py',
    'triangle_utils.py',
    '__init__.py',
    'SimReconstruction.py',
    'triangular_mesh.py',
    'triangle_mesh.py',
    'music.py',
    'octree.py',
    'setup.py',
    'dcimgSpoolShim.py',
    'labview_spooling_hacks.py',
    'dual_marching_cubes_v2.py',
    'spark.py',
    'dcimgFileChucker.py',
    'scanner_control.py',
    'pSpaceMap.py',
    'dcimg_chuck_test.py',
    'marching_cubes.py',
    'modified_marching_cubes.py',
    'isosurface.py',
    'marching_tetrahedra.py',
    'zMotionArtifactUtils.py',
    'pointFitConfoc.py',
)

py.install_sources(py_sources, subdir:'PYME/experimental')

data_files = files(
    '__init__.pxd',
    '_triangle_mesh.pxd',
    '_triangle_mesh.pyx',
    '_octree.pyx',
    'func_octree.pyx',
    '_treap.pyx',
)

install_data(data_files, install_dir: install_dir)

#FIXME - Extension()
#FIXME - Extension()
#FIXME - Extension()
#FIXME - Extension()
py.extension_module(
    '_triangle_mesh',
    [
        '_triangle_mesh.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/experimental',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
)

py.extension_module(
    '_octree',
    [
        '_octree.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/experimental',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
)
py.extension_module(
    '_treap',
    [
        '_treap.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/experimental',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
)
py.extension_module(
    'func_octree',
    [
        'func_octree.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/experimental',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
)

py.extension_module(
    'triangle_mesh_utils',
    [
        'triangle_mesh_utils.c',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/experimental',
    dependencies: py.dependency(),
    c_args: ['-O3', '-ffast-math'],
    install: true,
)