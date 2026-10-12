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


@pytest.mark.parametrize('sum_axis', [0, 1])
def test_peak_found_for_sum_axis(sum_axis):
    frame = _stripe_frame(40.3, stripe_axis=sum_axis)
    lock = _make_lock(frame, sum_axis=sum_axis)
    # default setpoint is the centre of the profile axis
    assert lock.peak_position == 0.5 * frame.shape[1 - sum_axis]

    lock.on_frame()
    assert lock._lock_ok
    assert lock.peak_position == pytest.approx(40.3, abs=0.05)

    lock.SetSubtractionProfile()
    assert lock.subtraction_profile.shape == (frame.shape[1 - sum_axis],)


def test_invalid_sum_axis():
    with pytest.raises(ValueError):
        _make_lock(_stripe_frame(40.), sum_axis=2)
