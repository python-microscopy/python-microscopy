
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/IO'

py_sources = files(
    'testClusterSpooling.py',
    'compatibility.py',
    'HTTPSpooler_v2.py',
    'clusterExport.py',
    'unifiedIO.py',
    'Spooler.py',
    'dcimg.py',
    'rgb_image.py',
    'MetaDataHandler.py',
    'clusterDuplication.py',
    'clusterIO.py',
    'csv_flavours.py',
    'tabular.py',
    'events.py',
    'clusterUpload.py',
    'clusterGlob.py',
    'acquisition_backends.py',
    'dataWrap.py',
    'dataExporter.py',
    '__init__.py',
    'ragged.py',
    'PZFFormat.py',
    'h5rFile.py',
    'clusterListing.py',
    'pyme_h5ls.py',
    'load_psf.py',
    'cluster_directory.py',
    'image.py',
    'h5File.py',
    'clusterResults.py',
    'cluster_streaming.py',
    'buffers.py',
)

py.install_sources(py_sources, subdir:'PYME/IO')

subdir('FileUtils')
py_sources = files(
    'DataSources/BioformatsDataSource.py',
    'DataSources/BaseDataSource.py',
    'DataSources/TiffDataSource.py',
    'DataSources/MultiviewDataSource.py',
    'DataSources/HTTPDataSource.py',
    'DataSources/SupertileDatasource.py',
    'DataSources/AlignDataSource.py',
    'DataSources/__init__.py',
    'DataSources/OBFDataSource.py',
    'DataSources/ArrayDataSource.py',
    'DataSources/UnsplitDataSource.py',
    'DataSources/ClusterPZFDataSource.py',
    'DataSources/CropDataSource.py',
    'DataSources/BGSDataSource.py',
    'DataSources/TQDataSource.py',
    'DataSources/ConcatenatedDataSource.py',
    'DataSources/RandomDataSource.py',
    'DataSources/BufferedDataSource.py',
    'DataSources/HDFDataSource.py',
    'DataSources/ElectronsPerSecondDataSource.py',
    'DataSources/ImageSeriesDataSource.py',
    'DataSources/FlatFieldDataSource.py',
    'DataSources/DcimgDataSource.py',
    'DataSources/TileDataSource.py',
)

py.install_sources(py_sources, subdir:'PYME/IO/DataSources')

subdir('countdir')

data_files = files(
    'buffer_helpers.pyx',
)

py.install_sources(data_files, subdir:'PYME/IO')
#FIXME - Extension()

py.extension_module(
    'buffer_helpers',
    [
        'buffer_helpers.pyx',
    ],
    include_directories: [np_include_dir],
    subdir: 'PYME/IO',
    dependencies: py.dependency(),
    c_args: ['-O3', '-fno-exceptions', '-ffast-math'],
    install: true,
    #cpp_args: ['-std=c++17'],
)