
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME'

py_sources = files(
    'update_version.py',
    'config.py',
    'version.py',
    'setup_exe.py',
    '__init__.py',
    'setup.py',
    'pyme_warnings.py',
)

py.install_sources(py_sources, subdir:'PYME')

subdir('Analysis')
subdir('Acquire')
subdir('DSView')
py_sources = files(
    'ParallelTasks/fitMon.py',
    'ParallelTasks/taskQueue.py',
    'ParallelTasks/taskWorkerZC.py',
    'ParallelTasks/setup_exe.py',
    'ParallelTasks/__init__.py',
    'ParallelTasks/launchWorkers.py',
    'ParallelTasks/taskServerZC.py',
    'ParallelTasks/fitMonP.py',
    'ParallelTasks/HDFTaskQueue.py',
    'ParallelTasks/DSTaskQueue.py',
)

py.install_sources(py_sources, subdir:'PYME')

subdir('IO')
py_sources = files(
    'Deconv/wiener.py',
    'Deconv/richardsonLucyMV.py',
    'Deconv/decThread.py',
    'Deconv/richardsonLucy.py',
    'Deconv/test_block_dec.py',
    'Deconv/beadGen.py',
    'Deconv/deconvDialogs.py',
    'Deconv/dec_test.py',
    'Deconv/__init__.py',
    'Deconv/dec.py',
    'Deconv/fftwWisdom.py',
    'Deconv/decTask.py',
    'Deconv/tq_block_dec.py',
    'Deconv/richardsonLucyMVM.py',
)

py.install_sources(py_sources, subdir:'PYME')

py_sources = files(
    'simulation/__init__.py',
    'simulation/wormlike.py',
    'simulation/wormlike2.py',
    'simulation/wormlike3.py',
    'simulation/pointsets.py',
    'simulation/wormlike_curled.py',
    'simulation/locify.py',
)

py.install_sources(py_sources, subdir:'PYME')

py_sources = files(
    'misc/pyme_zeroconf.py',
    'misc/big_sur_fix.py',
    'misc/hybrid_ns.py',
    'misc/fftw_compat.py',
    'misc/aniso.py',
    'misc/memUsage.py',
    'misc/facsPlot.py',
    'misc/eventSpy.py',
    'misc/sqlite_ns.py',
    'misc/zernike.py',
    'misc/sqlitendarray.py',
    'misc/snells.py',
    'misc/depGraph.py',
    'misc/findCycles.py',
    'misc/sffake.py',
    'misc/computerName.py',
    'misc/cython_numpy_monkey.py',
    'misc/locate.py',
    'misc/kdFit.py',
    'misc/__init__.py',
    'misc/memmap.py',
    'misc/colorsys.py',
    'misc/tempDB.py',
    'misc/read_agf.py',
    'misc/colormaps.py',
    'misc/editgrid.py',
    'misc/check_for_updates.py',
    'misc/fortran_interrupt_defeat.py',
    'misc/djangoRecarray.py',
    'misc/hillcurve.py',
    'misc/mock_ctypes.py',
    'misc/aligned_array.py',
    'misc/TiffImagePlugin.py',
    'misc/glassModel.py',
    'misc/coverslipGeometries.py',
    'misc/VolProj.py',
    'misc/hash32.py',
    'misc/exceptions.py',
    'misc/fbpIcons.py',
    'misc/wxThread.py',
    'misc/TiSettings.py',
    'misc/django_widgets.py',
    'misc/diskCleanup.py',
    'misc/mock_traitsui.py',
    'misc/dirSize.py',
    'misc/mock_traits.py',
    'misc/taskDef.py',
    'misc/pyro_tracebacks.py',
    'misc/zemax.py',
)

py.install_sources(py_sources, subdir:'PYME')

subdir('LMVis')
py_sources = files(
    'ui/patch_traitsui.py',
    'ui/MetadataTree.py',
    'ui/histLimits.py',
    'ui/filterPane.py',
    'ui/AUIFrame.py',
    'ui/wx_compat.py',
    'ui/shell.py',
    'ui/__init__.py',
    'ui/editFilterDialog.py',
    'ui/crop_dialog.py',
    'ui/mytimer.py',
    'ui/manualFoldPanel.py',
    'ui/fastGraph.py',
    'ui/editList.py',
    'ui/custom_traits_editors.py',
    'ui/selection.py',
    'ui/progress.py',
    'ui/autoFoldPanel.py',
    'ui/traitsui_constants.py',
    'ui/layerFoldPanel.py',
    'ui/recArrayView.py',
    'ui/cascading_layout.py',
    'ui/about_dlg.py',
)

py.install_sources(py_sources, subdir:'PYME')

py_sources = files(
    'util/log_verbosity.py',
    'util/webframework.py',
    'util/__init__.py',
    'util/pymelauncher.py',
    'util/install_dependencies.py',
    'util/execfile.py',
    'util/authenticate.py',
)

py.install_sources(py_sources, subdir:'PYME')

py_sources = files(
    'util/shmarray/shmarray.py',
    'util/shmarray/__init__.py',
    'util/shmarray/shmTest.py',
)

py.install_sources(py_sources, subdir:'PYME')

py_sources = files(
    'util/mProfile/__init__.py',
    'util/mProfile/tProfile.py',
    'util/mProfile/colorize_db_t.py',
    'util/mProfile/p.py',
    'util/mProfile/mProfile.py',
)

py.install_sources(py_sources, subdir:'PYME')

py_sources = files(
    'util/fProfile/viewProfile.py',
    'util/fProfile/convertProfile.py',
    'util/fProfile/__init__.py',
    'util/fProfile/fProfile.py',
)

py.install_sources(py_sources, subdir:'PYME')

subdir('localization')
subdir('recipes')
py_sources = files(
    'cluster/rulenodeserver.py',
    'cluster/HTTPRulePusher.py',
    'cluster/webdav.py',
    'cluster/dataserver_wsgi.py',
    'cluster/__init__.py',
    'cluster/rules.py',
    'cluster/PYMERuleServer.py',
    'cluster/distribution.py',
    'cluster/cluster_of_one.py',
    'cluster/HTTPTaskPusher.py',
    'cluster/taskWorkerHTTP.py',
    'cluster/HTTPDataServer.py',
    'cluster/ruleserver.py',
    'cluster/PYMERuleNodeServer.py',
    'cluster/status.py',
)

py.install_sources(py_sources, subdir:'PYME')

subdir('resources')
subdir('contrib')
subdir('experimental')
py_sources = files(
    'tileviewer/__init__.py',
    'tileviewer/tileviewer.py',
)

py.install_sources(py_sources, subdir:'PYME')

install_subdir('util/fProfile/html', install_dir: 'PYME/util/fProfile/html')
install_subdir('recipes/Recipes', install_dir: 'PYME/recipes/Recipes')
install_subdir('cluster/clusterUI', install_dir: 'PYME/cluster/clusterUI')
install_subdir('tileviewer/templates', install_dir: 'PYME/tileviewer/templates')
data_files = files(
    '__init__.pxd',
)

install_data(data_files, install_dir: 'PYME')
