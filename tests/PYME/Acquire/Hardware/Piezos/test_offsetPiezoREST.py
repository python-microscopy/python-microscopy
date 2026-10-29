"""Tests for offset trimming in PYME.Acquire.Hardware.Piezos.offsetPiezoREST."""
import pytest

from PYME.Acquire.Hardware.Piezos import offsetPiezoREST
from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase


class InterfaceOnlyPiezo(PiezoBase):
    """A piezo exposing only the PiezoBase interface.

    Notably it has no `max_travel` attribute - that is a convention of the older
    single-axis drivers, and is absent from e.g. SingleAxisWrapper. The limits
    are typically symmetric about zero, but might not be, e.g. it could
    report -200 to +200 um.
    """

    def __init__(self, min_pos=0.0, max_pos=100.0):
        self._min = min_pos
        self._max = max_pos
        self.position = 0.0

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        self.position = min(max(fPos, self._min), self._max)

    def GetPos(self, iChannel=0):
        return self.position

    def GetMin(self, iChan=1):
        return self._min

    def GetMax(self, iChan=1):
        return self._max


class _OffsetPiezo(offsetPiezoREST.OffsetPiezo):
    """OffsetPiezo with event logging stubbed out, so tests stay headless."""

    def LogFocusCorrection(self, offset):
        pass


@pytest.fixture
def offset_piezo():
    """Offset piezo over a 0 - 100 um base piezo, as the legacy drivers report."""
    return _OffsetPiezo(InterfaceOnlyPiezo(0.0, 100.0))


@pytest.fixture
def centred_offset_piezo():
    """Offset piezo over a base piezo reporting -200 to +200 um."""
    return _OffsetPiezo(InterfaceOnlyPiezo(-200.0, 200.0))


def test_correct_offset_without_max_travel_attribute(offset_piezo):
    """A base piezo with no `max_travel` should still work."""
    assert not hasattr(offset_piezo.basePiezo, 'max_travel')

    offset_piezo.basePiezo.MoveTo(0, 40.0)
    offset_piezo.CorrectOffset(5.0)

    assert offset_piezo.GetOffset() == pytest.approx(5.0)
    assert offset_piezo.basePiezo.GetPos() == pytest.approx(45.0)


def test_correct_offset_clamps_at_upper_limit(offset_piezo):
    offset_piezo.basePiezo.MoveTo(0, 90.0)
    offset_piezo.CorrectOffset(50.0)

    # base piezo target was 90, so only 10 um of correction is available
    assert offset_piezo.GetOffset() == pytest.approx(10.0)
    assert offset_piezo.basePiezo.GetPos() == pytest.approx(100.0)


def test_correct_offset_clamps_at_lower_limit(offset_piezo):
    offset_piezo.basePiezo.MoveTo(0, 20.0)
    offset_piezo.CorrectOffset(-50.0)

    assert offset_piezo.GetOffset() == pytest.approx(-20.0)
    assert offset_piezo.basePiezo.GetPos() == pytest.approx(0.0)


def test_get_max_offset_uses_interface(offset_piezo):
    offset_piezo.basePiezo.MoveTo(0, 30.0)

    assert offset_piezo.GetMaxOffset() == pytest.approx(70.0)
    assert offset_piezo.GetMinOffset() == pytest.approx(-30.0)


def test_centred_stage_can_use_negative_travel(centred_offset_piezo):
    """A base piezo centred on zero should be able to correct below zero.

    The clamp previously assumed a base piezo minimum of 0, which made the
    lower half of such a stage's travel unreachable.
    """
    centred_offset_piezo.basePiezo.MoveTo(0, 0.0)
    centred_offset_piezo.CorrectOffset(-150.0)

    assert centred_offset_piezo.GetOffset() == pytest.approx(-150.0)
    assert centred_offset_piezo.basePiezo.GetPos() == pytest.approx(-150.0)


def test_centred_stage_clamps_at_both_limits(centred_offset_piezo):
    centred_offset_piezo.basePiezo.MoveTo(0, 0.0)

    centred_offset_piezo.CorrectOffset(-500.0)
    assert centred_offset_piezo.basePiezo.GetPos() == pytest.approx(-200.0)

    centred_offset_piezo.CorrectOffset(1000.0)
    assert centred_offset_piezo.basePiezo.GetPos() == pytest.approx(200.0)


def test_centred_stage_offset_range(centred_offset_piezo):
    centred_offset_piezo.basePiezo.MoveTo(0, 50.0)

    assert centred_offset_piezo.GetMaxOffset() == pytest.approx(150.0)
    assert centred_offset_piezo.GetMinOffset() == pytest.approx(-250.0)
