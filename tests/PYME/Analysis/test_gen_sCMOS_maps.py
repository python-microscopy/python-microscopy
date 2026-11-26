import pytest

from PYME.IO.MetaDataHandler import NestedClassMDHandler
from PYME.Analysis.gen_sCMOS_maps import get_install_map_type


def _new_style_mdh(maptype, **kwargs):
    mdh = NestedClassMDHandler()
    mdh['CameraMap.Type'] = maptype
    for k, v in kwargs.items():
        mdh[k] = v
    return mdh


def _legacy_mdh(resultname, valid_width=2048, valid_height=2048, **kwargs):
    mdh = NestedClassMDHandler()
    mdh['Analysis.name'] = 'mean-variance'
    mdh['Analysis.resultname'] = resultname
    mdh['Analysis.valid.ROIWidth'] = valid_width
    mdh['Analysis.valid.ROIHeight'] = valid_height
    mdh['Camera.ROIWidth'] = 2048
    mdh['Camera.ROIHeight'] = 2048
    for k, v in kwargs.items():
        mdh[k] = v
    return mdh


@pytest.mark.parametrize('cmap_type, expected', [('mean', 'dark'), ('variance', 'variance')])
def test_camera_map_type(cmap_type, expected):
    assert get_install_map_type(_new_style_mdh(cmap_type)) == expected


@pytest.mark.parametrize('key', ['CameraMap.SubROI', 'CameraMap.Uniform'])
def test_camera_map_rejects_subroi_and_uniform(key):
    with pytest.raises(ValueError):
        get_install_map_type(_new_style_mdh('mean', **{key: True}))


def test_camera_map_unknown_type():
    with pytest.raises(ValueError):
        get_install_map_type(_new_style_mdh('flatfield'))


def test_camera_map_type_takes_precedence_over_legacy():
    mdh = _legacy_mdh('mean')
    mdh['CameraMap.Type'] = 'variance'
    assert get_install_map_type(mdh) == 'variance'


@pytest.mark.parametrize('resultname, expected', [('mean', 'dark'), ('variance', 'variance')])
def test_legacy_map_type(resultname, expected):
    assert get_install_map_type(_legacy_mdh(resultname)) == expected


@pytest.mark.parametrize('w, h', [(1024, 2048), (2048, 1024)])
def test_legacy_rejects_partial(w, h):
    with pytest.raises(ValueError):
        get_install_map_type(_legacy_mdh('mean', valid_width=w, valid_height=h))


def test_legacy_rejects_uniform():
    with pytest.raises(ValueError):
        get_install_map_type(_legacy_mdh('mean', **{'Analysis.isuniform': True}))


def test_not_a_map():
    with pytest.raises(ValueError):
        get_install_map_type(NestedClassMDHandler())
