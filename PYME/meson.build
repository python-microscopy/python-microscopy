
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME'
message('Installing PYME to: ' + install_dir)

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

# ParallelTasks (should now be replaced by PYMECluster)
# py_sources = files(
#     'ParallelTasks/fitMon.py',
#     'ParallelTasks/taskQueue.py',
#     'ParallelTasks/taskWorkerZC.py',
#     'ParallelTasks/setup_exe.py',
#     'ParallelTasks/__init__.py',
#     'ParallelTasks/launchWorkers.py',
#     'ParallelTasks/taskServerZC.py',
#     'ParallelTasks/fitMonP.py',
#     'ParallelTasks/HDFTaskQueue.py',
#     'ParallelTasks/DSTaskQueue.py',
# )

# py.install_sources(py_sources, subdir:'PYME/ParallelTasks')

subdir('IO')

#PYME.Deconv
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

py.install_sources(py_sources, subdir:'PYME/Deconv')

py_sources = files(
    'simulation/__init__.py',
    'simulation/wormlike.py',
    'simulation/wormlike2.py',
    'simulation/wormlike3.py',
    'simulation/pointsets.py',
    'simulation/wormlike_curled.py',
    'simulation/locify.py',
)

py.install_sources(py_sources, subdir:'PYME/simulation')

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

py.install_sources(py_sources, subdir:'PYME/misc')

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
py.install_sources(py_sources, subdir:'PYME/ui')

py_sources = files(
    'util/log_verbosity.py',
    'util/webframework.py',
    'util/__init__.py',
    'util/pymelauncher.py',
    'util/install_dependencies.py',
    'util/execfile.py',
    'util/authenticate.py',
)
py.install_sources(py_sources, subdir:'PYME/util')

py_sources = files(
    'util/shmarray/shmarray.py',
    'util/shmarray/__init__.py',
    'util/shmarray/shmTest.py',
)
py.install_sources(py_sources, subdir:'PYME/util/shmarray')

py_sources = files(
    'util/mProfile/__init__.py',
    'util/mProfile/tProfile.py',
    'util/mProfile/colorize_db_t.py',
    'util/mProfile/p.py',
    'util/mProfile/mProfile.py',
)
py.install_sources(py_sources, subdir:'PYME/util/mProfile')

py_sources = files(
    'util/fProfile/viewProfile.py',
    'util/fProfile/convertProfile.py',
    'util/fProfile/__init__.py',
    'util/fProfile/fProfile.py',
)
py.install_sources(py_sources, subdir:'PYME/util/fProfile')

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
py.install_sources(py_sources, subdir:'PYME/cluster')

subdir('resources')
subdir('contrib')
subdir('experimental')

py_sources = files(
    'tileviewer/__init__.py',
    'tileviewer/tileviewer.py',
)
py.install_sources(py_sources, subdir:'PYME/tileviewer')




fprofile_html_files = files(
    'util/fProfile/html/404.html',
    'util/fProfile/html/index.html',
    'util/fProfile/html/css/main.css',
    'util/fProfile/html/css/normalize.css',
    'util/fProfile/html/css/normalize.min.css',
    'util/fProfile/html/js/main.js',
    'util/fProfile/html/js/plugins.js',
    'util/fProfile/html/js/vendor/crossfilter.min.js',
    'util/fProfile/html/js/vendor/jquery-1.11.2.min.js',
    'util/fProfile/html/js/vendor/modernizr-2.8.3.min.js',
    'util/fProfile/html/js/vendor/tinycolor-min.js',
)
py.install_sources(fprofile_html_files, subdir:'PYME/', preserve_path:true)

recipes_files = files(
    'recipes/Recipes/STEDDeconvolution.yaml',
    'recipes/Recipes/STEDMinistacks.yaml',
    'recipes/Recipes/STED_COP3.yaml',
    'recipes/Recipes/blobMeas.yaml',
    'recipes/Recipes/blobMeas2.yaml',
    'recipes/Recipes/blobMeasSR.yaml',
    'recipes/Recipes/coloc.yaml',
    'recipes/Recipes/coloc_roi_masked.yaml',
    'recipes/Recipes/multiblob.yaml',
    'recipes/Recipes/pre_particleTracking.yaml',
    'recipes/Recipes/segmentAndMeasureLoci.yaml',
)
py.install_sources(recipes_files, subdir:'PYME/recipes/Recipes')



cluster_ui_files = files(
    'cluster/clusterUI/clusterui_tornado.py',
    'cluster/clusterUI/deploy.py',
    'cluster/clusterUI/manage.py',
    # accounts
    'cluster/clusterUI/accounts/__init__.py',
    'cluster/clusterUI/accounts/admin.py',
    'cluster/clusterUI/accounts/forms.py',
    'cluster/clusterUI/accounts/models.py',
    'cluster/clusterUI/accounts/urls.py',
    'cluster/clusterUI/accounts/views.py',
    'cluster/clusterUI/accounts/migrations/__init__.py',
    'cluster/clusterUI/accounts/migrations/0001_initial.py',
    'cluster/clusterUI/accounts/templates/accounts/user_form.html',
    'cluster/clusterUI/accounts/templates/registration/initial_complete.html',
    'cluster/clusterUI/accounts/templates/registration/initial_confirm.html',
    'cluster/clusterUI/accounts/templates/registration/initial_done.html',
    'cluster/clusterUI/accounts/templates/registration/verification.html',
    'cluster/clusterUI/accounts/templates/registration/verification_subject.txt',
    # clusterUI
    'cluster/clusterUI/clusterUI/__init__.py',
    'cluster/clusterUI/clusterUI/settings.py',
    'cluster/clusterUI/clusterUI/urls.py',
    'cluster/clusterUI/clusterUI/wsgi.py',
    # clusterbrowser
    'cluster/clusterUI/clusterbrowser/__init__.py',
    'cluster/clusterUI/clusterbrowser/admin.py',
    'cluster/clusterUI/clusterbrowser/apps.py',
    'cluster/clusterUI/clusterbrowser/models.py',
    'cluster/clusterUI/clusterbrowser/urls.py',
    'cluster/clusterUI/clusterbrowser/views.py',
    'cluster/clusterUI/clusterbrowser/migrations/__init__.py',
    'cluster/clusterUI/clusterbrowser/templates/base.html',
    'cluster/clusterUI/clusterbrowser/templates/clusterbrowser/dirlisting.html',
    'cluster/clusterUI/clusterbrowser/templates/clusterbrowser/form_new_folder.html',
    'cluster/clusterUI/clusterbrowser/templates/clusterbrowser/form_upload_file.html',
    'cluster/clusterUI/clusterbrowser/templates/clusterbrowser/lightlisting.html',
    'cluster/clusterUI/clusterbrowser/templates/registration/login.html',
    # clusterstatus
    'cluster/clusterUI/clusterstatus/__init__.py',
    'cluster/clusterUI/clusterstatus/admin.py',
    'cluster/clusterUI/clusterstatus/apps.py',
    'cluster/clusterUI/clusterstatus/models.py',
    'cluster/clusterUI/clusterstatus/urls.py',
    'cluster/clusterUI/clusterstatus/views.py',
    'cluster/clusterUI/clusterstatus/migrations/__init__.py',
    'cluster/clusterUI/clusterstatus/templates/clusterstatus/load_content.html',
    'cluster/clusterUI/clusterstatus/templates/clusterstatus/queue_info.html',
    'cluster/clusterUI/clusterstatus/templates/clusterstatus/queue_info_content.html',
    'cluster/clusterUI/clusterstatus/templates/clusterstatus/status_dash.html',
    'cluster/clusterUI/clusterstatus/templates/clusterstatus/status_dash_content.html',
    # localization
    'cluster/clusterUI/localization/__init__.py',
    'cluster/clusterUI/localization/admin.py',
    'cluster/clusterUI/localization/apps.py',
    'cluster/clusterUI/localization/models.py',
    'cluster/clusterUI/localization/urls.py',
    'cluster/clusterUI/localization/views.py',
    'cluster/clusterUI/localization/migrations/__init__.py',
    'cluster/clusterUI/localization/templates/localization/settings_form.html',
    # recipes
    'cluster/clusterUI/recipes/__init__.py',
    'cluster/clusterUI/recipes/admin.py',
    'cluster/clusterUI/recipes/models.py',
    'cluster/clusterUI/recipes/urls.py',
    'cluster/clusterUI/recipes/views.py',
    'cluster/clusterUI/recipes/migrations/__init__.py',
    'cluster/clusterUI/recipes/templates/recipes/extra_inputs.html',
    'cluster/clusterUI/recipes/templates/recipes/form_recipe.html',
    'cluster/clusterUI/recipes/templates/recipes/input_list.html',
    'cluster/clusterUI/recipes/templates/recipes/recipe_editor.html',
    'cluster/clusterUI/recipes/templates/recipes/recipe_editor_common.html',
    'cluster/clusterUI/recipes/templates/recipes/recipe_editrun.html',
    'cluster/clusterUI/recipes/templates/recipes/recipe_standalone.html',
    'cluster/clusterUI/recipes/templates/recipes/recipe_template.html',
)
py.install_sources(cluster_ui_files, subdir:'PYME', preserve_path:true)

tileviewer_templates_files = files(
    'tileviewer/templates/roi_list.html',
    'tileviewer/templates/tileviewer.html',
)
py.install_sources(tileviewer_templates_files, subdir:'PYME/tileviewer/templates')

# data_files = files(
#     '__init__.pxd',
# )

# install_data(data_files, install_dir: install_dir)

