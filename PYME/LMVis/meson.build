
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
py = import('python').find_installation(pure: false)
np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/LMVis'

py_sources = files(
    'displayPane.py',
    'glcanvas_core.py',
    'pointSettingsPanel.py',
    'imageView.py',
    'rendGauss.py',
    'visCore.py',
    'dyeRatios.py',
    'gl_render3D.py',
    'objectDarkMeasure.py',
    'triBlobs.py',
    'visHelpersMin.py',
    'genImageDialog.py',
    'view_clipping_pane.py',
    'layer_panel.py',
    'sessionpaths.py',
    'mv_math.py',
    'workspaceTree.py',
    'setup_exe.py',
    'importTextDialog.py',
    '__init__.py',
    'renderers.py',
    'setup.py',
    'gl_render.py',
    'statusLog.py',
    'pipeline.py',
    'quadTreeSettings.py',
    'imageView2.py',
    'progGraph.py',
    'colourFilterGUI.py',
    'visHelpers.py',
    'colourPanel.py',
    'VisGUI.py',
    'gl_render3D_shaders.py',
    'pipelineView.py',
    'views.py',
    'gl_offScreenHandler.py',
)

py.install_sources(py_sources, subdir:'PYME/LMVis')

py_sources = files(
    'Extras/spherical_harmonics.py',
    'Extras/annotation.py',
    'Extras/particleTracking.py',
    'Extras/sequentialSpecies.py',
    'Extras/extra_layers.py',
    'Extras/fusionUtils.py',
    'Extras/vibration.py',
    'Extras/clusterAnalysis.py',
    'Extras/setBGImage.py',
    'Extras/flagZMotionArtifacts.py',
    'Extras/QPobjectSegment.py',
    'Extras/__init__.py',
    'Extras/pointwiseColoc.py',
    'Extras/multiColourAnalysis.py',
    'Extras/pointSetGeneration.py',
    'Extras/animation.py',
    'Extras/dockedPanel.py',
    'Extras/scaleBox.py',
    'Extras/perFrameVariable.py',
    'Extras/temperaturePlot.py',
    'Extras/multiviewMapping.py',
    'Extras/objectMeasurements.py',
    'Extras/photophysics.py',
    'Extras/surface_fitting.py',
    'Extras/saved_views.py',
    'Extras/subsets.py',
    'Extras/showShortcuts.py',
    'Extras/shiftmapGenerator.py',
    'Extras/fiducial.py',
    'Extras/snapshot.py',
)

py.install_sources(py_sources, subdir:'PYME/LMVis')

py_sources = files(
    'layers/labels.py',
    'layers/quiver.py',
    'layers/tracks.py',
    'layers/OverlayLayer.py',
    'layers/image_layer.py',
    'layers/SelectionOverlayLayer.py',
    'layers/layer_defaults.py',
    'layers/ScaleBarOverlayLayer.py',
    'layers/ScaleBoxOverlayLayer.py',
    'layers/__init__.py',
    'layers/mesh.py',
    'layers/octree.py',
    'layers/pointcloud.py',
    'layers/text.py',
    'layers/AxesOverlayLayer.py',
    'layers/LUTOverlayLayer.py',
    'layers/base.py',
)

py.install_sources(py_sources, subdir:'PYME/LMVis')

py_sources = files(
    'shader_programs/shader_program.py',
    'shader_programs/DefaultShaderProgram.py',
    'shader_programs/PointSpriteShaderProgram.py',
    'shader_programs/shader_loader.py',
    'shader_programs/__init__.py',
    'shader_programs/TesselShaderProgram.py',
    'shader_programs/WireFrameShaderProgram.py',
    'shader_programs/GouraudShaderProgram.py',
    'shader_programs/GLProgram.py',
    'shader_programs/oit_compositor.py',
    'shader_programs/ShaderProgramFactory.py',
)

py.install_sources(py_sources, subdir:'PYME/LMVis')

install_subdir('shader_programs/shaders', install_dir: 'PYME/LMVis/shader_programs/shaders')
