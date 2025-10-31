
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/resources'

py_sources = files(
    '__init__.py',
)

py.install_sources(py_sources, subdir:'PYME/resources')


# icons
icon_files = files(
    'icons/LUT.png',
    'icons/YSM_Shield.png',
    'icons/YSM_logo.png',
    'icons/arrow.png',
    'icons/crosshairs.png',
    'icons/line_select.png',
    'icons/logo_fmhs.png',
    'icons/media-playback-pause.png',
    'icons/media-playback-start.png',
    'icons/media-skip-backward.png',
    'icons/media-skip-forward.png',
    'icons/mesh.png',
    'icons/mesh.svg',
    'icons/pmacquire.icns',
    'icons/pmacquire.ico',
    'icons/pmanal.icns',
    'icons/pmanal.ico',
    'icons/pmvis.icns',
    'icons/pmvis.ico',
    'icons/points.png',
    'icons/points.svg',
    'icons/pymeLogo.png',
    'icons/rect_select.png',
    'icons/squiggle_select.png',
    'icons/text3800.icns',
    'icons/text3800.ico',
    'icons/text3800.png',
    'icons/text3800.tif',
    'icons/text5377.icns',
    'icons/text5377.ico',
    'icons/text5377.png',
    'icons/tracks.png',
    'icons/tracks.svg',
    'icons/view-front.png',
    'icons/view-home.png',
    'icons/view-right.png',
    'icons/view-top.png',
)
py.install_sources(icon_files, subdir:'PYME/resources/icons')

# config_template
config_template_files = files(
    'config_template/distributor.yaml',
    'config_template/nodeserver.yaml',
)
py.install_sources(config_template_files, subdir:'PYME/resources/config_template')

# web
web_files = files(
    'web/rule_inline.html',
    'web/ruleserver.html',
)
py.install_sources(web_files, subdir:'PYME/resources/web')

# web/static
web_static_files = files(
    'web/static/bootstrap.min.js',
    'web/static/cluster.css',
    'web/static/dashboard.css',
    'web/static/fileselector.js',
    'web/static/jigna-vue-bare.js',
    'web/static/jquery.min.js',
    'web/static/jstree.min.js',
    'web/static/openSeadragonGL.js',
    'web/static/openseadragon-svg-overlay.js',
    'web/static/openseadragon-viewerinputhook.min.js',
    'web/static/openseadragon.min.js',
    'web/static/recipe_editor.js',
    'web/static/utils.js',
    'web/static/viaWebGL.js',
)
py.install_sources(web_static_files, subdir:'PYME/resources/web/static')

# web/static/css
web_static_css_files = files(
    'web/static/css/bootstrap-theme.css',
    'web/static/css/bootstrap-theme.min.css',
    'web/static/css/bootstrap.css',
    'web/static/css/bootstrap.min.css',
    'web/static/css/flat_design_colors_full.css',
    'web/static/css/recipe_editor.css',
)
py.install_sources(web_static_css_files, subdir:'PYME/resources/web/static/css')

# web/static/fonts
web_static_fonts_files = files(
    'web/static/fonts/glyphicons-halflings-regular.eot',
    'web/static/fonts/glyphicons-halflings-regular.svg',
    'web/static/fonts/glyphicons-halflings-regular.ttf',
    'web/static/fonts/glyphicons-halflings-regular.woff',
    'web/static/fonts/glyphicons-halflings-regular.woff2',
)
py.install_sources(web_static_fonts_files, subdir:'PYME/resources/web/static/fonts')

# web/static/images/osd_flat
web_static_images_osd_flat_files = files(
    'web/static/images/osd_flat/LICENSE.txt',
    'web/static/images/osd_flat/README.md',
    'web/static/images/osd_flat/fullpage_grouphover.png',
    'web/static/images/osd_flat/fullpage_hover.png',
    'web/static/images/osd_flat/fullpage_pressed.png',
    'web/static/images/osd_flat/fullpage_rest.png',
    'web/static/images/osd_flat/home_grouphover.png',
    'web/static/images/osd_flat/home_hover.png',
    'web/static/images/osd_flat/home_pressed.png',
    'web/static/images/osd_flat/home_rest.png',
    'web/static/images/osd_flat/next_grouphover.png',
    'web/static/images/osd_flat/next_hover.png',
    'web/static/images/osd_flat/next_pressed.png',
    'web/static/images/osd_flat/next_rest.png',
    'web/static/images/osd_flat/previous_grouphover.png',
    'web/static/images/osd_flat/previous_hover.png',
    'web/static/images/osd_flat/previous_pressed.png',
    'web/static/images/osd_flat/previous_rest.png',
    'web/static/images/osd_flat/rotateleft_grouphover.png',
    'web/static/images/osd_flat/rotateleft_hover.png',
    'web/static/images/osd_flat/rotateleft_pressed.png',
    'web/static/images/osd_flat/rotateleft_rest.png',
    'web/static/images/osd_flat/rotateright_grouphover.png',
    'web/static/images/osd_flat/rotateright_hover.png',
    'web/static/images/osd_flat/rotateright_pressed.png',
    'web/static/images/osd_flat/rotateright_rest.png',
    'web/static/images/osd_flat/zoomin_grouphover.png',
    'web/static/images/osd_flat/zoomin_hover.png',
    'web/static/images/osd_flat/zoomin_pressed.png',
    'web/static/images/osd_flat/zoomin_rest.png',
    'web/static/images/osd_flat/zoomout_grouphover.png',
    'web/static/images/osd_flat/zoomout_hover.png',
    'web/static/images/osd_flat/zoomout_pressed.png',
    'web/static/images/osd_flat/zoomout_rest.png',
)
py.install_sources(web_static_images_osd_flat_files, subdir:'PYME/resources/web/static/images/osd_flat')

# web/static/shaders/fragment
web_static_shaders_fragment_files = files(
    'web/static/shaders/fragment/lut.glsl',
    'web/static/shaders/fragment/none.glsl',
    'web/static/shaders/fragment/outline.glsl',
    'web/static/shaders/fragment/rect.glsl',
    'web/static/shaders/fragment/sobel3.glsl',
    'web/static/shaders/fragment/white.glsl',
)
py.install_sources(web_static_shaders_fragment_files, subdir:'PYME/resources/web/static/shaders/fragment')

# web/static/shaders/vertex
web_static_shaders_vertex_files = files(
    'web/static/shaders/vertex/rect.glsl',
    'web/static/shaders/vertex/square.glsl',
)
py.install_sources(web_static_shaders_vertex_files, subdir:'PYME/resources/web/static/shaders/vertex')

# test_datasets
test_datasets_files = files(
    'test_datasets/README.md',
    'test_datasets/t_im.tif',
    'test_datasets/test_csv.csv',
    'test_datasets/test_hdf.hdf',
)
py.install_sources(test_datasets_files, subdir:'PYME/resources/test_datasets')
