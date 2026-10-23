"""Tests for endpoint registration in PYME.util.webframework."""
import pytest

from PYME.util import webframework


class _Endpoints(object):
    """Stand-in for a server class mixing endpoints with ordinary properties."""

    def __init__(self):
        self.property_evaluations = 0

    @property
    def cheap_property(self):
        self.property_evaluations += 1
        return 42

    @property
    def broken_property(self):
        self.property_evaluations += 1
        raise RuntimeError('this property cannot be evaluated in this context')

    @webframework.register_endpoint('/do_thing')
    def do_thing(self):
        return 'done'

    def not_an_endpoint(self):
        return None


def _collect_endpoints(obj, prefix=''):
    """Run add_endpoints against a minimal holder, avoiding socket setup."""
    class _Holder(object):
        pass

    holder = _Holder()
    holder._endpoints = {}
    webframework.APIHTTPServer.add_endpoints(holder, obj, prefix)
    return holder._endpoints


def test_add_endpoints_registers_bound_method():
    obj = _Endpoints()
    endpoints = _collect_endpoints(obj)

    assert list(endpoints.keys()) == ['/do_thing']
    # the registered callable should be bound, i.e. callable with no arguments
    assert endpoints['/do_thing']() == 'done'


def test_add_endpoints_honours_prefix():
    endpoints = _collect_endpoints(_Endpoints(), prefix='/thing_doer')

    assert list(endpoints.keys()) == ['/thing_doer/do_thing']


def test_add_endpoints_does_not_evaluate_properties():
    """Scanning for endpoints must not have side effects.

    Properties on hardware server classes can be expensive, or can raise if the
    hardware is not in a suitable state - neither should affect registration.
    """
    obj = _Endpoints()
    endpoints = _collect_endpoints(obj)

    assert obj.property_evaluations == 0
    assert '/do_thing' in endpoints


def test_add_endpoints_survives_raising_property():
    obj = _Endpoints()

    # sanity check that the property really does raise
    with pytest.raises(RuntimeError):
        obj.broken_property

    assert '/do_thing' in _collect_endpoints(obj)
