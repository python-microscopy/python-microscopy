import numpy as np
import pytest
from PYME.IO.image import ImageStack
from PYME.recipes import modules  # ensure all recipe modules are registered
from PYME.recipes.base import ExtractChannel


def _make_multichannel_imagestack(nx=16, ny=12, nz=3, nt=2, nc=4):
    """Create a 5D (X, Y, Z, T, C) ImageStack where each channel c has a
    constant pixel value of (c + 1) * 10"""
    data = np.zeros((nx, ny, nz, nt, nc), dtype='float32')
    for c in range(nc):
        data[:, :, :, :, c] = (c + 1) * 10.0
    return ImageStack(data)


class TestExtractChannelWith5DSource:
    """Regression tests for ExtractChannel against multi-channel 5D data.

    Previously, passing a 5D datasource (shape X, Y, Z, T, C) would cause
    _pickChannel to use the deprecated image.data accessor, which returned a
    4D (X, Y, Z*T, C) view — the 4 channels appeared as Z-slices and all of
    them were returned instead of just the requested one."""

    def setup_method(self):
        self.nc = 4
        self.nz = 3
        self.nt = 2
        self.image = _make_multichannel_imagestack(nc=self.nc, nz=self.nz, nt=self.nt)

    def test_input_is_recognised_as_multichannel(self):
        """Sanity check: the input image should be seen as nc channels."""
        assert self.image.data_xyztc.shape[4] == self.nc

    @pytest.mark.parametrize("chan", [0, 1, 2, 3])
    def test_extracted_image_has_one_channel(self, chan):
        mod = ExtractChannel(channelToExtract=chan)
        out = mod.apply_simple(self.image)
        assert out.data_xyztc.shape[4] == 1, (
            f"Expected 1 channel in output, got {out.data_xyztc.shape[4]} "
            f"(channelToExtract={chan})"
        )

    @pytest.mark.parametrize("chan", [0, 1, 2, 3])
    def test_extracted_channel_preserves_z_and_t(self, chan):
        mod = ExtractChannel(channelToExtract=chan)
        out = mod.apply_simple(self.image)
        shape = out.data_xyztc.shape
        assert shape[2] == self.nz, f"Z dimension changed: expected {self.nz}, got {shape[2]}"
        assert shape[3] == self.nt, f"T dimension changed: expected {self.nt}, got {shape[3]}"

    @pytest.mark.parametrize("chan", [0, 1, 2, 3])
    def test_extracted_channel_data_is_correct(self, chan):
        """The pixel values of the extracted channel must match those of the
        corresponding input channel, not any other channel."""
        mod = ExtractChannel(channelToExtract=chan)
        out = mod.apply_simple(self.image)
        expected_value = (chan + 1) * 10.0
        pixel_values = out.data_xyztc[:, :, :, :, 0]
        assert np.allclose(pixel_values, expected_value), (
            f"Channel {chan}: expected all pixels == {expected_value}, "
            f"got min={pixel_values.min()}, max={pixel_values.max()}"
        )
