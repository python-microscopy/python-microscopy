import numpy as np
import pytest
from unittest import mock

pytest.importorskip('simple_pid')

from PYME.Acquire.Hardware.focus_locks.reflection_focus_lock import ReflectedLinePIDFocusLock


def _stripe_frame(peak, shape=(100, 128), sigma=3., stripe_axis=0):
    """Frame with a Gaussian stripe extending along `stripe_axis`."""
    n_profile = shape[1 - stripe_axis]
    line = 1000 * np.exp(-0.5 * ((np.arange(n_profile) - peak) / sigma) ** 2) + 10
    frame = np.broadcast_to(line, (shape[stripe_axis], n_profile))
    if stripe_axis == 1:
        frame = frame.T
    return np.ascontiguousarray(frame)[:, :, None]


def _make_lock(frame, **kwargs):
    scope = mock.MagicMock(spec=['frameWrangler'])
    scope.frameWrangler.currentFrame = frame
    piezo = mock.MagicMock()
    piezo.GetMax.return_value = 100.
    piezo.GetMin.return_value = 0.
    return ReflectedLinePIDFocusLock(scope, piezo, trigger_failsafe=False, **kwargs)


def test_toggle_lock_switches_both_ways():
    """ToggleLock must be able to turn the lock off again.

    A bitwise not was used here previously - ~True is -2 and ~False is -1, both
    truthy, so the lock latched on.
    """
    lock = _make_lock(_stripe_frame(40.))
    assert not lock.LockEnabled()

    lock.ToggleLock()
    assert lock.LockEnabled()

    lock.ToggleLock()
    assert not lock.LockEnabled()


def test_toggle_lock_logs_offset_when_disabling():
    """Disabling via ToggleLock should go through EnableLock's bookkeeping."""
    lock = _make_lock(_stripe_frame(40.))
    lock.ToggleLock()
    lock.piezo.LogFocusCorrection.assert_not_called()

    lock.ToggleLock()
    lock.piezo.LogFocusCorrection.assert_called_once()


def test_enable_lock_handles_string_false():
    """`enable` arrives as a string over the REST interface."""
    lock = _make_lock(_stripe_frame(40.))
    lock.EnableLock(True)
    assert lock.LockEnabled()

    lock.EnableLock('False')
    assert not lock.LockEnabled()
