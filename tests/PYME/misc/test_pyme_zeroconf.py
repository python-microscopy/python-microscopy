from PYME.misc.pyme_zeroconf import _uses_new_service_info_api


def test_uses_new_service_info_api_with_non_lexicographic_version():
    assert _uses_new_service_info_api('0.150.0')


def test_uses_new_service_info_api_at_version_boundary():
    assert _uses_new_service_info_api('0.35.0')
    assert not _uses_new_service_info_api('0.34.9')
    assert not _uses_new_service_info_api('0.35.0rc1')
