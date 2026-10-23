import pytest

from PYME.Acquire import microscope


class Scope:
    pass


def test_get_permitted_values_defaults_to_any_value():
    state = microscope.StateManager(Scope())
    state.registerHandler('test.key')

    assert microscope.ANY_VALUE is state.get_permitted_values('test.key')


def test_get_permitted_values_returns_registered_values():
    permitted_values = ['first', 'second']
    state = microscope.StateManager(Scope())
    state.registerHandler('test.key', permitted_values=permitted_values)

    assert state.get_permitted_values('test.key') is permitted_values


def test_get_permitted_values_raises_for_unregistered_key():
    state = microscope.StateManager(Scope())

    with pytest.raises(KeyError):
        state.get_permitted_values('test.key')
